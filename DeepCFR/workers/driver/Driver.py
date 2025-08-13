from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.driver._HighLevelAlgo import HighLevelAlgo
from PokerRL._.TensorboardLogger import TensorboardLogger
from PokerRL.rl.MaybeRay import MaybeRay
from PokerRL.rl.base_cls.workers.DriverBase import DriverBase
import os
import socket
import subprocess
import psutil
import ray
import torch
import re
import shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Driver(DriverBase):

    def __init__(self, t_prof, eval_methods, n_iterations=None, iteration_to_import=None, name_to_import=None):
        if t_prof.DISTRIBUTED:
            from DeepCFR.workers.chief.dist import Chief
            from DeepCFR.workers.la.dist import LearnerActor
            from DeepCFR.workers.ps.dist import ParameterServer
        else:
            from DeepCFR.workers.chief.local import Chief
            from DeepCFR.workers.la.local import LearnerActor
            from DeepCFR.workers.ps.local import ParameterServer

        total_cpu = psutil.cpu_count() or 1
        desired_actors = t_prof.n_learner_actors + t_prof.n_seats + len(eval_methods) + 1  # +1 for Chief
        cpu_fraction = total_cpu / desired_actors
        MaybeRay._default_num_cpus = cpu_fraction

        total_mem = psutil.virtual_memory().total
        ray_mem = min(2 * (10 ** 10), int(total_mem * 0.8))
        memory_per_la = ray_mem / max(1, t_prof.n_learner_actors)

        def _is_cuda(device):
            return (
                (isinstance(device, torch.device) and device.type == "cuda")
                or (isinstance(device, str) and device.startswith("cuda"))
            )

        total_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

        la_uses_gpu = _is_cuda(t_prof.module_args["adv_training"].device_training) or _is_cuda(t_prof.device_inference)
        ps_uses_gpu = _is_cuda(t_prof.device_parameter_server)
        eval_uses_gpu = _is_cuda(t_prof.device_inference)

        la_gpu_workers = t_prof.n_learner_actors if la_uses_gpu else 0
        ps_gpu_workers = t_prof.n_seats if ps_uses_gpu else 0
        eval_gpu_workers = len(eval_methods) if eval_uses_gpu else 0
        gpu_workers = la_gpu_workers + ps_gpu_workers + eval_gpu_workers
        gpu_fraction = min(1.0, total_gpu / gpu_workers) if gpu_workers > 0 else 0

        MaybeRay._default_num_gpus = gpu_fraction if eval_uses_gpu else 0

        super().__init__(t_prof=t_prof, eval_methods=eval_methods, n_iterations=n_iterations,
                         iteration_to_import=iteration_to_import, name_to_import=name_to_import,
                         chief_cls=Chief, eval_agent_cls=EvalAgentDeepCFR)

        # Determine Ray's log directory and configure TensorBoard
        #
        # Ray has changed a number of its internal APIs over time.  Older
        # versions exposed ``ray._private.worker.global_node`` while newer
        # releases renamed the attribute to ``_global_node`` and removed the
        # helper ``get_ray_temp_dir``.  We try a best-effort sequence of
        # lookups that works across a wide range of Ray versions without
        # depending on any specific private function.
        try:
            # Ray >= 1.7 used ``_global_node``.  In some versions the public
            # ``get_logs_dir`` is still available.
            ray_log_root = ray._private.worker._global_node.get_logs_dir()
        except Exception:
            # Fall back to session_dir/logs which is available on newer Ray
            # versions.  If this also fails, use Ray's temp directory utility
            # (``get_temp_dir``) or finally ``tempfile.gettempdir``.
            try:
                session_dir = getattr(ray._private.worker._global_node, "address_info", {}).get("session_dir")
                if session_dir:
                    ray_log_root = os.path.join(session_dir, "logs")
                else:
                    raise RuntimeError("session_dir not available")
            except Exception:
                try:
                    from ray._private.utils import get_temp_dir

                    ray_log_root = get_temp_dir()
                except Exception:
                    import tempfile

                    ray_log_root = tempfile.gettempdir()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_name = re.sub(r"[^\w.-]", "_", str(t_prof.name))
        run_root = os.path.join(ray_log_root, "tensorboard", sanitized_name)
        if os.path.exists(run_root):
            archive_root = os.path.join(run_root, "archive")
            os.makedirs(archive_root, exist_ok=True)
            for entry in os.listdir(run_root):
                full = os.path.join(run_root, entry)
                if full == archive_root:
                    continue
                if os.path.isdir(full):
                    shutil.move(full, os.path.join(archive_root, entry))
        t_prof.path_log_storage = os.path.join(run_root, timestamp)
        os.makedirs(t_prof.path_log_storage, exist_ok=True)

        if getattr(t_prof, "tb_writer", None) is not None:
            try:
                t_prof.tb_writer.close()
            except Exception:
                pass
        t_prof.tb_writer = SummaryWriter(log_dir=t_prof.path_log_storage) if t_prof.log_verbose else None
        if t_prof.log_verbose:
            print(f"TensorBoard logs will be written to {t_prof.path_log_storage}")

        # Recreate logger with updated path
        self.logger = TensorboardLogger(
            name=t_prof.name,
            chief_handle=self.chief_handle,
            path_log_storage=t_prof.path_log_storage,
            runs_distributed=t_prof.DISTRIBUTED,
            runs_cluster=t_prof.CLUSTER,
        )

        def _get_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        tb_port = _get_free_port()
        tb_cmd = [
            "tensorboard",
            "--logdir",
            t_prof.path_log_storage,
            "--host",
            "0.0.0.0",
            "--port",
            str(tb_port),
        ]
        try:
            self._tb_proc = subprocess.Popen(
                tb_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            self._tb_port = tb_port
            print(f"TensorBoard listening on http://0.0.0.0:{tb_port}/")
        except Exception as ex:
            self._tb_proc = None
            print(f"Failed to start TensorBoard: {ex}")

        self._cpu_fraction = cpu_fraction
        self._gpu_fraction = gpu_fraction
        self._la_uses_gpu = la_uses_gpu
        self._ps_uses_gpu = ps_uses_gpu
        self._memory_per_la = memory_per_la

        if "h2h" in list(eval_methods.keys()):
            assert EvalAgentDeepCFR.EVAL_MODE_SINGLE in t_prof.eval_modes_of_algo
            assert EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in t_prof.eval_modes_of_algo
            self._ray.remote(self.eval_masters["h2h"][0].set_modes,
                             [EvalAgentDeepCFR.EVAL_MODE_SINGLE, EvalAgentDeepCFR.EVAL_MODE_AVRG_NET]
                             )

        print("Creating LAs...")
        self.la_handles = [
            self._ray.create_worker(LearnerActor,
                                    t_prof,
                                    i,
                                    self.chief_handle,
                                    num_gpus=self._gpu_fraction if self._la_uses_gpu else 0,
                                    num_cpus=self._cpu_fraction,
                                    memory=self._memory_per_la)
            for i in range(t_prof.n_learner_actors)
        ]

        print("Creating Parameter Servers...")
        self.ps_handles = [
            self._ray.create_worker(
                ParameterServer,
                t_prof,
                p,
                self.chief_handle,
                num_gpus=self._gpu_fraction if self._ps_uses_gpu else 0,
                num_cpus=self._cpu_fraction,
            )
            for p in range(t_prof.n_seats)
        ]

        self._ray.wait([
            self._ray.remote(self.chief_handle.set_ps_handle,
                             *self.ps_handles),
            self._ray.remote(self.chief_handle.set_la_handles,
                             *self.la_handles)
        ])

        print("Created and initialized Workers")

        self.algo = HighLevelAlgo(t_prof=t_prof,
                                  la_handles=self.la_handles,
                                  ps_handles=self.ps_handles,
                                  chief_handle=self.chief_handle)

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        self._maybe_load_checkpoint_init()

    def run(self):
        print("Setting stuff up...")

        # """"""""""""""""
        # Init globally
        # """"""""""""""""
        self.algo.init()

        print("Starting Training...")
        try:
            for _iter_nr in range(10000000 if self.n_iterations is None else self.n_iterations):
                print("Iteration: ", self._cfr_iter)
    
                # """"""""""""""""
                # Maybe train AVRG
                # """"""""""""""""
                avrg_times = None
                if self._AVRG and self._any_eval_needs_avrg_net():
                    avrg_times = self.algo.train_average_nets(cfr_iter=_iter_nr)
    
                # """"""""""""""""
                # Eval
                # """"""""""""""""
                # Evaluate. Sync & Lock, then train while evaluating on other workers
                self.evaluate()
    
                # """"""""""""""""
                # Log
                # """"""""""""""""
                if self._cfr_iter % self._t_prof.log_export_freq == 0:
                    self.save_logs()
                    self._show_log_dir_usage()
                self.periodically_export_eval_agent()
    
                # """"""""""""""""
                # Iteration
                # """"""""""""""""
                iter_times = self.algo.run_one_iter_alternating_update(cfr_iter=self._cfr_iter)
    
                print(
                    "Generating Data: ", str(iter_times["t_generating_data"]) + "s.",
                    "  ||  Trained ADV", str(iter_times["t_computation_adv"]) + "s.",
                    "  ||  Synced ADV", str(iter_times["t_syncing_adv"]) + "s.",
                    "\n"
                )
                if self._AVRG and avrg_times:
                    print(
                        "Trained AVRG", str(avrg_times["t_computation_avrg"]) + "s.",
                        "  ||  Synced AVRG", str(avrg_times["t_syncing_avrg"]) + "s.",
                        "\n"
                    )
    
                self._cfr_iter += 1
    
                # """"""""""""""""
                # Checkpoint
                # """"""""""""""""
                self.periodically_checkpoint()
        finally:
            if getattr(self, "_tb_proc", None):
                self._tb_proc.terminate()
                try:
                    self._tb_proc.wait(timeout=5)
                except Exception:
                    pass

    def _any_eval_needs_avrg_net(self):
        for e in list(self.eval_masters.values()):
            if self._cfr_iter % e[1] == 0:
                return True
        return False

    def _show_log_dir_usage(self):
        if not os.path.exists(self._t_prof.path_log_storage):
            return
        try:
            subprocess.run(["ls", "-lh", self._t_prof.path_log_storage], check=False)
        except Exception as ex:
            print(f"Failed to list log directory: {ex}")

    def checkpoint(self, **kwargs):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]:
            self._ray.wait([
                self._ray.remote(w.checkpoint,
                                 self._cfr_iter)
            ])

        # Delete past checkpoints
        s = [self._cfr_iter]
        if self._cfr_iter > self._t_prof.checkpoint_freq + 1:
            s.append(self._cfr_iter - self._t_prof.checkpoint_freq)

        self._delete_past_checkpoints(steps_not_to_delete=s)

    def load_checkpoint(self, step, name_to_load):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]:
            self._ray.wait([
                self._ray.remote(w.load_checkpoint,
                                 name_to_load, step)
            ])
