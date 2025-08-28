import os
import pickle
import sys
from types import SimpleNamespace

import pytest

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from DeepCFR.workers.chief import local


class _UnpickleableWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass

    def __getstate__(self):  # pragma: no cover - should never be called
        raise RuntimeError("Writer should not be pickled")


class _DummyStrategyBuffer:
    def __init__(self, *args, **kwargs):
        self.size = 0
        self.owner = kwargs.get("owner")

    def add(self, iteration_strat):
        self.size += 1

    def get(self, i):
        class _Dummy:
            def state_dict(self):
                return {}

        return _Dummy()

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass


def test_chief_is_pickleable_without_writers(monkeypatch, tmp_path):
    # Patch external dependencies to keep the Chief lightweight
    monkeypatch.setattr(local.rl_util, "get_env_builder", lambda t_prof: object())
    monkeypatch.setattr(local, "StrategyBuffer", _DummyStrategyBuffer)
    monkeypatch.setattr(local, "SummaryWriter", _UnpickleableWriter)
    monkeypatch.setattr(local, "resolve_device", lambda _: "cpu")

    t_prof = SimpleNamespace(
        DISTRIBUTED=False,
        CLUSTER=False,
        eval_modes_of_algo=["SINGLE"],
        log_verbose=True,
        n_seats=2,
        device_inference="cpu",
        module_args={"env": None},
        env_builder_cls_str="",
        game_cls_str="",
        path_log_storage=str(tmp_path),
        name="test",
    )

    chief = local.Chief(t_prof)
    # _writers should still be empty after initialization
    assert chief._writers == {}

    # Creating an experiment populates _writers with an unpickleable object
    chief.create_experiment("exp")
    assert chief._writers

    data = pickle.dumps(chief)
    restored = pickle.loads(data)
    assert restored._writers == {}

