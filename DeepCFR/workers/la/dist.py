import ray

from DeepCFR.workers.la.local import LearnerActor as LocalLearnerActor


@ray.remote
class LearnerActor(LocalLearnerActor):

    def __init__(self, t_prof, worker_id, chief_handle):
        LocalLearnerActor.__init__(self, t_prof=t_prof, worker_id=worker_id, chief_handle=chief_handle)
