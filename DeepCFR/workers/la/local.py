import os
import pickle
import time

import psutil
import torch
import logging

from DeepCFR.IterationStrategy import IterationStrategy
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.la.buffers.AdvReservoirBuffer import AdvReservoirBuffer
from DeepCFR.workers.la.AdvWrapper import AdvWrapper
from DeepCFR.workers.la.buffers.AvrgReservoirBuffer import AvrgReservoirBuffer
from DeepCFR.workers.la.AvrgWrapper import AvrgWrapper
from DeepCFR.workers.la.sampling_algorithms.MultiOutcomeSampler import MultiOutcomeSampler
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase
from DeepCFR.utils.device import resolve_device


class LearnerActor(WorkerBase):

    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof)

        self._adv_args = t_prof.module_args["adv_training"]

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        self._id = worker_id
        self._chief_handle = chief_handle

        self._device_inference = resolve_device(self._t_prof.device_inference)
        self._adv_device = resolve_device(self._adv_args.device_training)

        self._adv_buffers = [
            AdvReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._adv_args.max_buffer_size,
                               nn_type=t_prof.nn_type,
                               iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
            for p in range(self._t_prof.n_seats)
        ]

        self._adv_wrappers = [
            AdvWrapper(owner=p,
                       env_bldr=self._env_bldr,
                       adv_training_args=self._adv_args,
                       device=self._adv_device)
            for p in range(self._t_prof.n_seats)
        ]

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]
            self._avrg_device = resolve_device(self._avrg_args.device_training)

            self._avrg_buffers = [
                AvrgReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._avrg_args.max_buffer_size,
                                    nn_type=t_prof.nn_type,
                                    iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
                for p in range(self._t_prof.n_seats)
            ]

            self._avrg_wrappers = [
                AvrgWrapper(owner=p,
                            env_bldr=self._env_bldr,
                            avrg_training_args=self._avrg_args,
                            device=self._avrg_device)
                for p in range(self._t_prof.n_seats)
            ]

            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=self._avrg_buffers,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples)
            else:
                raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")
        else:
            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=None,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples)
            else:
                raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")

        self._gpu_device = None
        devices = [self._adv_device]
        if self._AVRG:
            devices.append(self._avrg_device)
        for dev in devices:
            if dev.type == "cuda":
                self._gpu_device = dev
                break

        self._gpu_metrics_available = True
        if self._gpu_device is not None:
            try:
                torch.cuda.memory_reserved(self._gpu_device)
                torch.cuda.utilization(self._gpu_device)
            except Exception as e:
                logging.warning(f"GPU metrics unavailable: {e}")
                self._gpu_metrics_available = False

        self._perf_log_interval = 10
        self._perf_metrics = {
            "generate_data": {"time": 0.0, "cpu": 0.0, "gpu_mem": 0.0, "gpu_util": 0.0, "count": 0, "total": 0},
            "update": {"time": 0.0, "cpu": 0.0, "gpu_mem": 0.0, "gpu_util": 0.0, "count": 0, "total": 0},
        }

        if self._t_prof.log_verbose:
            self._exp_perf_handle = self._ray.get(
                self._ray.remote(self._chief_handle.create_experiment,
                                 f"LA{worker_id}/Perf"))
            self._exp_mem_usage_handle = self._ray.get(
                self._ray.remote(self._chief_handle.create_experiment,
                                 f"LA{worker_id}/Memory_Usage"))
            self._exps_adv_buffer_size_handles = self._ray.get(
                [
                    self._ray.remote(
                        self._chief_handle.create_experiment,
                        f"LA{worker_id}/P{p}/ADV_BufSize",
                    )
                    for p in range(self._t_prof.n_seats)
                ]
            )
            if self._AVRG:
                self._exps_avrg_buffer_size_handles = self._ray.get(
                    [
                        self._ray.remote(
                            self._chief_handle.create_experiment,
                            f"LA{worker_id}/P{p}/AVRG_BufSize",
                        )
                        for p in range(self._t_prof.n_seats)
                    ]
                )

    def generate_data(self, traverser, cfr_iter):
        process = psutil.Process(os.getpid())
        process.cpu_percent()
        t_start = time.perf_counter()

        iteration_strats = [
            IterationStrategy(t_prof=self._t_prof, env_bldr=self._env_bldr, owner=p,
                              device=self._device_inference, cfr_iter=cfr_iter)
            for p in range(self._t_prof.n_seats)
        ]
        for s in iteration_strats:
            s.load_net_state_dict(state_dict=self._adv_wrappers[s.owner].net_state_dict())

        self._data_sampler.generate(n_traversals=self._t_prof.n_traversals_per_iter,
                                    traverser=traverser,
                                    iteration_strats=iteration_strats,
                                    cfr_iter=cfr_iter,
                                    )

        duration = time.perf_counter() - t_start
        cpu = process.cpu_percent()
        gpu_mem = gpu_util = 0.0
        if self._gpu_device is not None and self._gpu_metrics_available:
            try:
                gpu_mem = torch.cuda.memory_reserved(self._gpu_device)
                gpu_util = torch.cuda.utilization(self._gpu_device)
            except Exception as e:
                logging.warning(f"Failed to query GPU metrics: {e}")
                self._gpu_metrics_available = False

        m = self._perf_metrics["generate_data"]
        m["time"] += duration
        m["cpu"] += cpu
        m["gpu_mem"] += gpu_mem
        m["gpu_util"] += gpu_util
        m["count"] += 1
        m["total"] += 1
        if self._t_prof.log_verbose and m["count"] >= self._perf_log_interval:
            avg_time = m["time"] / m["count"]
            avg_cpu = m["cpu"] / m["count"]
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_perf_handle, "GenerateData/Time", m["total"], avg_time)
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_perf_handle, "GenerateData/CPU", m["total"], avg_cpu)
            if self._gpu_device is not None and self._gpu_metrics_available:
                avg_mem = m["gpu_mem"] / m["count"]
                avg_util = m["gpu_util"] / m["count"]
                self._ray.remote(self._chief_handle.add_scalar,
                                 self._exp_perf_handle, "GenerateData/GPUMem", m["total"], avg_mem)
                self._ray.remote(self._chief_handle.add_scalar,
                                 self._exp_perf_handle, "GenerateData/GPUUtil", m["total"], avg_util)
            m.update({"time": 0.0, "cpu": 0.0, "gpu_mem": 0.0, "gpu_util": 0.0, "count": 0})

        # Log after both players generated data
        if self._t_prof.log_verbose and traverser == 1 and (cfr_iter % 3 == 0):
            for p in range(self._t_prof.n_seats):
                self._ray.remote(self._chief_handle.add_scalar,
                                 self._exps_adv_buffer_size_handles[p], "Debug/BufferSize", cfr_iter,
                                 self._adv_buffers[p].size)
                if self._AVRG:
                    self._ray.remote(self._chief_handle.add_scalar,
                                     self._exps_avrg_buffer_size_handles[p], "Debug/BufferSize", cfr_iter,
                                     self._avrg_buffers[p].size)

            process = psutil.Process(os.getpid())
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_mem_usage_handle, "Debug/MemoryUsage/LA", cfr_iter,
                             process.memory_info().rss)

    def update(self, adv_state_dicts=None, avrg_state_dicts=None):
        """
        Args:
            adv_state_dicts (list):         Optional. if not None:
                                                        expects a list of neural net state dicts or None for each player
                                                        in order of their seat_ids. This allows updating only some
                                                        players.

            avrg_state_dicts (list):         Optional. if not None:
                                                        expects a list of neural net state dicts or None for each player
                                                        in order of their seat_ids. This allows updating only some
                                                        players.
        """
        process = psutil.Process(os.getpid())
        process.cpu_percent()
        t_start = time.perf_counter()

        for p_id in range(self._t_prof.n_seats):
            if adv_state_dicts[p_id] is not None:
                self._adv_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(self._ray.get(adv_state_dicts[p_id]),
                                                             device=self._adv_wrappers[p_id].device))

            if avrg_state_dicts[p_id] is not None:
                self._avrg_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(self._ray.get(avrg_state_dicts[p_id]),
                                                             device=self._avrg_wrappers[p_id].device))

        duration = time.perf_counter() - t_start
        cpu = process.cpu_percent()
        gpu_mem = gpu_util = 0.0
        if self._gpu_device is not None and self._gpu_metrics_available:
            try:
                gpu_mem = torch.cuda.memory_reserved(self._gpu_device)
                gpu_util = torch.cuda.utilization(self._gpu_device)
            except Exception as e:
                logging.warning(f"Failed to query GPU metrics: {e}")
                self._gpu_metrics_available = False

        m = self._perf_metrics["update"]
        m["time"] += duration
        m["cpu"] += cpu
        m["gpu_mem"] += gpu_mem
        m["gpu_util"] += gpu_util
        m["count"] += 1
        m["total"] += 1
        if self._t_prof.log_verbose and m["count"] >= self._perf_log_interval:
            avg_time = m["time"] / m["count"]
            avg_cpu = m["cpu"] / m["count"]
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_perf_handle, "Update/Time", m["total"], avg_time)
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_perf_handle, "Update/CPU", m["total"], avg_cpu)
            if self._gpu_device is not None and self._gpu_metrics_available:
                avg_mem = m["gpu_mem"] / m["count"]
                avg_util = m["gpu_util"] / m["count"]
                self._ray.remote(self._chief_handle.add_scalar,
                                 self._exp_perf_handle, "Update/GPUMem", m["total"], avg_mem)
                self._ray.remote(self._chief_handle.add_scalar,
                                 self._exp_perf_handle, "Update/GPUUtil", m["total"], avg_util)
            m.update({"time": 0.0, "cpu": 0.0, "gpu_mem": 0.0, "gpu_util": 0.0, "count": 0})

    def get_loss_last_batch_adv(self, p_id):
        return self._adv_wrappers[p_id].loss_last_batch

    def get_loss_last_batch_avrg(self, p_id):
        return self._avrg_wrappers[p_id].loss_last_batch

    def get_adv_grads(self, p_id):
        return self._ray.grads_to_numpy(
            self._adv_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._adv_buffers[p_id]))

    def get_avrg_grads(self, p_id):
        return self._ray.grads_to_numpy(
            self._avrg_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._avrg_buffers[p_id]))

    def checkpoint(self, curr_step):
        for p_id in range(self._env_bldr.N_SEATS):
            state = {
                "adv_buffer": self._adv_buffers[p_id].state_dict(),
                "adv_wrappers": self._adv_wrappers[p_id].state_dict(),
                "p_id": p_id,
            }
            if self._AVRG:
                state["avrg_buffer"] = self._avrg_buffers[p_id].state_dict()
                state["avrg_wrappers"] = self._avrg_wrappers[p_id].state_dict()

            with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "wb") as pkl_file:
                pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        for p_id in range(self._env_bldr.N_SEATS):
            with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "rb") as pkl_file:
                state = pickle.load(pkl_file)

                assert state["p_id"] == p_id

                self._adv_buffers[p_id].load_state_dict(state["adv_buffer"])
                self._adv_wrappers[p_id].load_state_dict(state["adv_wrappers"])
                if self._AVRG:
                    self._avrg_buffers[p_id].load_state_dict(state["avrg_buffer"])
                    self._avrg_wrappers[p_id].load_state_dict(state["avrg_wrappers"])
