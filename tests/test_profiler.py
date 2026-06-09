import sys
import types

sys.modules.setdefault("tensorflow", types.SimpleNamespace(lite=types.SimpleNamespace()))

from src.profiler import ModelProfiler


def test_profiler_reports_file_size_and_latency(monkeypatch, tmp_path):
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"0" * 2048)

    profiler = ModelProfiler()
    monkeypatch.setattr(profiler, "_measure_inference_time", lambda path: 12.34567)

    metrics = profiler.profile_model(str(model_path))

    assert metrics == {"size_mb": 0.002, "latency_ms": 12.3457}