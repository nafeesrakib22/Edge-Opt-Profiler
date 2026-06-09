import pandas as pd

from src.ranker import ModelRanker


def test_ranker_prefers_smallest_model_when_size_weight_dominates():
    df = pd.DataFrame(
        [
            {"model": "large", "size_mb": 10.0, "latency_ms": 5.0, "accuracy": 0.95},
            {"model": "small", "size_mb": 2.0, "latency_ms": 8.0, "accuracy": 0.90},
        ]
    )

    ranked = ModelRanker().rank_models(df, alpha=1.0, beta=0.0, gamma=0.0)

    assert ranked.iloc[0]["model"] == "small"


def test_ranker_handles_equal_metric_ranges_without_division_error():
    df = pd.DataFrame(
        [
            {"model": "a", "size_mb": 1.0, "latency_ms": 5.0, "accuracy": 0.90},
            {"model": "b", "size_mb": 1.0, "latency_ms": 5.0, "accuracy": 0.95},
        ]
    )

    ranked = ModelRanker().rank_models(df, alpha=0.3, beta=0.4, gamma=0.3)

    assert "final_score" in ranked.columns
    assert ranked["final_score"].notna().all()