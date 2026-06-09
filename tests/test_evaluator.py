import sys
import types

sys.modules.setdefault("tensorflow", types.SimpleNamespace(lite=types.SimpleNamespace()))

from src.evaluator import AccuracyEngine


def test_evaluator_simulates_expected_variant_accuracy():
    evaluator = AccuracyEngine()

    assert evaluator.get_accuracy("unused.tflite", "baseline") == 1.00
    assert evaluator.get_accuracy("unused.tflite", "fp16") == 0.999
    assert evaluator.get_accuracy("unused.tflite", "dynamic") == 0.965
    assert evaluator.get_accuracy("unused.tflite", "pruning") == 0.95