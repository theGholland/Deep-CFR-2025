"""Deep Counterfactual Regret Minimization algorithms."""

__all__ = []
__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Compatibility patch for Ray >=2.0
# ---------------------------------------------------------------------------
#
# The original PokerRL `MaybeRay` wrapper passed a `redis_max_memory` argument
# when initialising Ray.  This option was removed from newer Ray releases,
# causing `ray.init` to raise `RuntimeError: Unknown keyword argument(s)` when
# running Deep-CFR with modern Ray versions.  Here we monkey‑patch the
# `init_local` method to drop the obsolete parameter and keep the existing
# behaviour of limiting the object store memory.
#
# This code executes on import so the patched behaviour is applied before the
# framework initialises Ray.
try:  # pragma: no cover - best effort patching
    import psutil
    import ray
    import numpy as np
    from PokerRL.rl.MaybeRay import MaybeRay

    def _init_local(self):
        if self.runs_distributed:
            # Auto-detect available resources so Ray can schedule workers on both
            # CPU and GPU without additional configuration.  This prevents Ray
            # from crashing when CUDA is present but resources have not been
            # declared explicitly.
            total_mem = psutil.virtual_memory().total
            obj_mem = getattr(MaybeRay, "_object_store_memory", None)
            if obj_mem is None:
                obj_mem = int(total_mem * 0.4)
            elif isinstance(obj_mem, float) and 0 < obj_mem < 1:
                obj_mem = int(total_mem * obj_mem)
            ray_init_kwargs = {
                # Allow configuration of Ray's object store memory via
                # ``TrainingProfile``.  If unset, reserve roughly 40% of system
                # RAM for the object store so that the remainder stays
                # available for actors and tasks.  Passing a float between 0
                # and 1 will be interpreted as a fraction of total memory.
                "object_store_memory": int(obj_mem),
                "num_cpus": psutil.cpu_count() or 1,
                # Ray 2.x already exposes a built-in ``memory`` resource for
                # scheduling and no longer allows configuring it via the
                # ``resources`` dict.  We therefore rely on Ray's default
                # behaviour, which advertises the machine's total available
                # memory.  Previously this code attempted to limit worker
                # memory to ~80% of the system RAM; this caused crashes with
                # newer Ray versions.  Removing it restores compatibility.
                # Enable Ray's web dashboard so users can monitor resource usage
                # and task progress at http://localhost:8265.  We bind to
                # ``0.0.0.0`` so the dashboard is reachable when running on a
                # remote machine.
                "include_dashboard": True,
                "dashboard_host": "0.0.0.0",
            }

            ray.init(**ray_init_kwargs)
            try:  # pragma: no cover - best effort GPU detection
                # Detect presence of GPUs but do not reserve them by default.
                # Callers must explicitly request GPU resources when creating
                # actors so that multiple workers can share the available
                # devices without contention.
                ray.cluster_resources().get("GPU", 0)
            finally:
                MaybeRay._default_num_gpus = 0
    MaybeRay.init_local = _init_local

    # Default CPU fraction per worker.  This can be overridden at runtime by
    # setting ``MaybeRay._default_num_cpus`` before creating any actors.
    MaybeRay._default_num_cpus = 1
    MaybeRay._default_num_gpus = 0
    MaybeRay._object_store_memory = None

    def _create_worker(self, cls, *args, num_gpus=0, num_cpus=None, memory=None):
        """Create a Ray actor with optional resource allocation.

        GPU resources are not reserved automatically.  Actors that require a
        GPU must specify ``num_gpus`` explicitly when created.
        """
        if self.runs_distributed:
            if num_cpus is None:
                num_cpus = getattr(self, "_default_num_cpus", 1)
            options_kwargs = {"num_gpus": num_gpus, "num_cpus": num_cpus}
            if memory is not None:
                options_kwargs["memory"] = memory
            return cls.options(**options_kwargs).remote(*args)
        return cls(*args)

    MaybeRay.create_worker = _create_worker

    def _state_dict_to_torch(self, _dict, device):
        """Convert numpy arrays in a state dict to writable torch tensors."""
        import torch

        new_dict = {}
        if self.runs_distributed:
            for k in list(_dict.keys()):
                if isinstance(_dict[k], np.ndarray):
                    arr = _dict[k]
                    if not arr.flags.writeable:
                        arr = arr.copy()
                    new_dict[k] = torch.from_numpy(arr)
                else:
                    new_dict[k] = _dict[k]

                new_dict[k] = new_dict[k].to(device)
        else:
            for k in list(_dict.keys()):
                new_dict[k] = _dict[k].to(device)

        return new_dict

    MaybeRay.state_dict_to_torch = _state_dict_to_torch

    # Ensure actors created during DriverBase initialisation do not reserve GPU
    # resources.  DriverBase spawns the Chief and various evaluation helpers
    # without specifying resource requirements, causing them to inherit the
    # current default GPU fraction.  We temporarily force this default to zero
    # while the base class sets up these CPU-only workers.
    from PokerRL.rl.base_cls.workers.DriverBase import DriverBase

    _orig_driverbase_init = DriverBase.__init__

    def _driverbase_init(self, t_prof, eval_methods, chief_cls, eval_agent_cls,
                         n_iterations=None, iteration_to_import=None,
                         name_to_import=None):
        prev_default = getattr(MaybeRay, "_default_num_gpus", 0)
        MaybeRay._default_num_gpus = 0
        try:
            return _orig_driverbase_init(
                self,
                t_prof,
                eval_methods,
                chief_cls,
                eval_agent_cls,
                n_iterations,
                iteration_to_import,
                name_to_import,
            )
        finally:  # pragma: no cover - best effort restoration
            MaybeRay._default_num_gpus = prev_default

    DriverBase.__init__ = _driverbase_init
except Exception:  # noqa: S110
    # Dependencies are optional at import time; if they are missing we simply
    # skip the patch and rely on the original implementation.
    pass

