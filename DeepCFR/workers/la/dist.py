import ray

from DeepCFR.workers.la.local import LearnerActor as LocalLearnerActor


@ray.remote
class LearnerActor(LocalLearnerActor):
    """Distributed ``LearnerActor`` that rebuilds the Chief handle from a name."""

    def __init__(self, t_prof, worker_id, chief_ref):
        super().__init__(t_prof=t_prof, worker_id=worker_id, chief_ref=chief_ref)
