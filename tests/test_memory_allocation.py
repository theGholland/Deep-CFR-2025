import os
import sys

import ray
from ray._private import utils as ray_utils
from types import SimpleNamespace

# Ensure project root is on path for direct module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from DeepCFR.workers.driver.Driver import Driver

def test_memory_allocation_respects_cgroup_limit(monkeypatch):
    def fake_cluster_resources():
        raise RuntimeError("no resources")
    monkeypatch.setattr(ray, "cluster_resources", fake_cluster_resources)
    monkeypatch.setattr(ray_utils, "get_system_memory", lambda: 1 * 1024 ** 3)
    t_prof = SimpleNamespace(
        memory_per_worker=None,
        memory_per_worker_multiplier=1.0,
        n_learner_actors=1,
        n_seats=1,
    )
    mem = Driver._calc_memory_per_worker(t_prof)
    expected_total = 1 * 1024 ** 3
    expected_ray_mem = min(2 * (10 ** 10), int(expected_total * 0.8))
    assert mem == int(expected_ray_mem / 2)
