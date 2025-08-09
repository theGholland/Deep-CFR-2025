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
    from PokerRL.rl.MaybeRay import MaybeRay

    def _init_local(self):
        if self.runs_distributed:
            # Auto-detect available resources so Ray can schedule workers on both
            # CPU and GPU without additional configuration.  This prevents Ray
            # from crashing when CUDA is present but resources have not been
            # declared explicitly.
            ray_init_kwargs = {
                "object_store_memory": min(
                    2 * (10 ** 10), int(psutil.virtual_memory().total * 0.4)
                ),
                "num_cpus": psutil.cpu_count() or 1,
            }

            try:
                if torch.cuda.is_available():
                    ray_init_kwargs["num_gpus"] = torch.cuda.device_count()
            except Exception:  # pragma: no cover - best effort GPU detection
                pass

            ray.init(**ray_init_kwargs)

    MaybeRay.init_local = _init_local
except Exception:  # noqa: S110
    # Dependencies are optional at import time; if they are missing we simply
    # skip the patch and rely on the original implementation.
    pass

