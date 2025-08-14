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
    import torch
    import numpy as np
    from PokerRL.rl.MaybeRay import MaybeRay

    def _init_local(self):
        if self.runs_distributed:
            # Auto-detect available resources so Ray can schedule workers on both
            # CPU and GPU without additional configuration.  This prevents Ray
            # from crashing when CUDA is present but resources have not been
            # declared explicitly.
            total_mem = psutil.virtual_memory().total
            ray_init_kwargs = {
                # Reserve only 20% of RAM for Ray's object store so that roughly
                # 80% remains available for actors and tasks.  Using a
                # percentage instead of a fixed value adapts to different
                # machine sizes automatically.
                "object_store_memory": int(total_mem * 0.2),
                "num_cpus": psutil.cpu_count() or 1,
                # Advertise only ~80% of total RAM to Ray's scheduler so worker
                # processes cannot collectively exceed that amount.
                "resources": {"memory": int(total_mem * 0.8)},
                # Enable Ray's web dashboard so users can monitor resource usage
                # and task progress at http://localhost:8265.  We bind to
                # ``0.0.0.0`` so the dashboard is reachable when running on a
                # remote machine.
                "include_dashboard": True,
                "dashboard_host": "0.0.0.0",
            }

            try:
                if torch.cuda.is_available():
                    ray_init_kwargs["num_gpus"] = torch.cuda.device_count()
            except Exception:  # pragma: no cover - best effort GPU detection
                pass

            ray.init(**ray_init_kwargs)
    MaybeRay.init_local = _init_local

    # Default CPU fraction per worker.  This can be overridden at runtime by
    # setting ``MaybeRay._default_num_cpus`` before creating any actors.
    MaybeRay._default_num_cpus = 1
    MaybeRay._default_num_gpus = 0

    def _create_worker(self, cls, *args, num_gpus=None, num_cpus=None, memory=None):
        """Create a Ray actor with optional resource allocation."""
        if self.runs_distributed:
            if num_gpus is None:
                try:
                    num_gpus = getattr(self, "_default_num_gpus", 0)
                except Exception:  # pragma: no cover - best effort GPU detection
                    num_gpus = 0
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
except Exception:  # noqa: S110
    # Dependencies are optional at import time; if they are missing we simply
    # skip the patch and rely on the original implementation.
    pass

