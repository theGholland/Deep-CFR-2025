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
import warnings
from datetime import datetime
from DeepCFR.utils.device import resolve_device


class Driver(DriverBase):

    @staticmethod
    def _get_total_memory():
        """Return total system memory in bytes.

        Prefers Ray's reported resources but falls back to Ray's
        ``get_system_memory`` utility which respects cgroup limits.  Only if
        both mechanisms fail do we consult ``psutil`` as a last resort.
        """
        try:
            total_mem = int(ray.cluster_resources().get("memory", 0))
            if total_mem > 0:
                return total_mem
        except Exception:
            pass
        try:
            from ray._private.utils import get_system_memory

            total_mem = int(get_system_memory())
            if total_mem > 0:
                return total_mem
        except Exception:
            pass
        return int(psutil.virtual_memory().total)

    @classmethod
    def _calc_memory_per_worker(cls, t_prof):
        if t_prof.memory_per_worker == 0:
            return None
        if t_prof.memory_per_worker is None:
            total_mem = cls._get_total_memory()
            ray_mem = min(2 * (10 ** 10), int(total_mem * 0.8))
            n_mem_workers = t_prof.n_learner_actors + t_prof.n_seats
            memory_per_worker = int(ray_mem / max(1, n_mem_workers))
        else:
            memory_per_worker = int(t_prof.memory_per_worker)
        memory_per_worker = int(memory_per_worker * t_prof.memory_per_worker_multiplier)
        if memory_per_worker <= 0:
            return None
        return memory_per_worker

    def __init__(self, t_prof, eval_methods, n_iterations=None, iteration_to_import=None, name_to_import=None):
        if t_prof.DISTRIBUTED:
            from DeepCFR.workers.chief.dist import Chief
            from DeepCFR.workers.la.dist import LearnerActor
            from DeepCFR.workers.ps.dist import ParameterServer
        else:
            from DeepCFR.workers.chief.local import Chief
            from DeepCFR.workers.la.local import LearnerActor
            from DeepCFR.workers.ps.local import ParameterServer

        try:
            total_cpu = int(ray.cluster_resources().get("CPU", 0))
        except Exception:
            total_cpu = 0
        if total_cpu <= 0:
            try:
                total_cpu = len(os.sched_getaffinity(0))
            except Exception:
                total_cpu = 1

        desired_actors = t_prof.n_learner_actors + t_prof.n_seats + len(eval_methods) + 1  # +1 for Chief
        cpu_fraction = total_cpu / max(1, desired_actors)
        MaybeRay._default_num_cpus = cpu_fraction

        # Memory reservation per Ray worker.  If ``memory_per_worker`` is 0 we
        # omit the reservation entirely.  Otherwise we either compute a default
        # from available RAM or use the explicit value, scaling by
        # ``memory_per_worker_multiplier`` to support larger models.
        memory_per_worker = self._calc_memory_per_worker(t_prof)
        memory_per_la = memory_per_worker
        memory_per_ps = memory_per_worker

        # Determine Ray's log directory and compute the TensorBoard log path
        # before calling ``super().__init__`` so that any base-class
        # initialization that depends on ``t_prof.path_log_storage`` sees the
        # finalized value.
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

        # Respect an existing log directory if the TrainingProfile already
        # specified one.  This allows callers to control where TensorBoard
        # writes its event files.  Only fall back to Ray's temporary logging
        # directory when no path has been set yet.
        if not getattr(t_prof, "path_log_storage", None):
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

        super().__init__(t_prof=t_prof, eval_methods=eval_methods, n_iterations=n_iterations,
                         iteration_to_import=iteration_to_import, name_to_import=name_to_import,
                         chief_cls=Chief, eval_agent_cls=EvalAgentDeepCFR)

        def _is_cuda(device):
            return (
                (isinstance(device, torch.device) and device.type == "cuda")
                or (isinstance(device, str) and device.startswith("cuda"))
            )

        adv_spec = t_prof.module_args["adv_training"].device_training
        inf_spec = t_prof.device_inference
        ps_spec = t_prof.device_parameter_server

        any_cuda_requested = any(
            _is_cuda(d) for d in (adv_spec, inf_spec, ps_spec)
        )

        if any_cuda_requested:
            try:
                total_gpu = ray.cluster_resources().get("GPU", 0)
            except Exception:
                total_gpu = 0
            try:
                cuda_available = torch.cuda.is_available()
            except Exception:
                cuda_available = False
            ray_has_gpu = total_gpu > 0

            def _resolve(spec):
                if isinstance(spec, str) and spec.lower() == "auto":
                    spec = "cuda" if (cuda_available and ray_has_gpu) else "cpu"
                return resolve_device(spec)
        else:
            total_gpu = 0
            cuda_available = False
            ray_has_gpu = False

            def _resolve(_):
                return torch.device("cpu")

        adv_device = _resolve(adv_spec)
        inf_device = _resolve(inf_spec)
        ps_device = _resolve(ps_spec)

        def _ensure_device(dev, name):
            if _is_cuda(dev) and (not cuda_available or not ray_has_gpu):
                warnings.warn(
                    f"CUDA device requested for {name} but GPUs are unavailable; falling back to CPU.",
                    RuntimeWarning,
                )
                return torch.device("cpu")
            return dev

        adv_device = _ensure_device(adv_device, "adv_training")
        inf_device = _ensure_device(inf_device, "inference")
        ps_device = _ensure_device(ps_device, "parameter_server")

        la_uses_gpu = _is_cuda(adv_device) or _is_cuda(inf_device)
        ps_uses_gpu = _is_cuda(ps_device)
        eval_uses_gpu = _is_cuda(inf_device)

        la_gpu_workers = t_prof.n_learner_actors if la_uses_gpu else 0
        ps_gpu_workers = t_prof.n_seats if ps_uses_gpu else 0
        eval_gpu_workers = len(eval_methods) if eval_uses_gpu else 0
        gpu_workers = la_gpu_workers + ps_gpu_workers + eval_gpu_workers
        gpu_fraction = total_gpu / max(1, gpu_workers)

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

        self._tb_proc = None
        if t_prof.log_verbose:
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
        self._memory_per_ps = memory_per_ps

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
                memory=self._memory_per_ps,
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
        except RuntimeError as e:
            print(f"Training stopped: {e}")
        finally:
            try:
                self._ray.wait([self._ray.remote(self.chief_handle.flush_tb_writers)])
                self._ray.wait([self._ray.remote(self.chief_handle.close_tb_writers)])
            except Exception:
                pass

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
