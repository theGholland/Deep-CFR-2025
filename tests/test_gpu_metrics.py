import types
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from DeepCFR.workers.la import local


def test_check_gpu_metrics_available_with_utilization(monkeypatch):
    fake_cuda = types.SimpleNamespace(
        memory_reserved=lambda device: 1,
        utilization=lambda device: 2,
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(local, "torch", fake_torch)

    available, use_nvidia_smi = local._check_gpu_metrics_available(object())
    assert available is True
    assert use_nvidia_smi is False


def test_check_gpu_metrics_available_without_utilization(monkeypatch):
    fake_cuda = types.SimpleNamespace()
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(local, "torch", fake_torch)

    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(local.subprocess, "run", fake_run)

    available, use_nvidia_smi = local._check_gpu_metrics_available(object())
    assert available is False
    assert use_nvidia_smi is False
