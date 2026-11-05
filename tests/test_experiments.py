import json
import pytest

from experiments.run import run_experiment


def config():
    return {"seeds": [7], "tasks": ["corrupted_observation", "legitimate_large", "arithmetic"],
            "methods": ["mul", "hubermul", "clipped_mul", "normalized_mul", "grad_clip", "huber_loss"],
            "steps": 3, "eval_every": 1, "train_size": 24, "validation_size": 16,
            "test_size": 16, "contamination": 0.1, "outlier_scale": 5.0,
            "extrapolation_scale": 3.0, "projection_bound": 2.0,
            "success_threshold": 0.05,
            "candidates": [{"learning_rate": 0.01, "parameter": 2.0}]}


def test_runner_records_and_reproduces(tmp_path):
    a = run_experiment(config(), tmp_path / "a")
    b = run_experiment(config(), tmp_path / "b")
    assert a == b
    assert len(a["results"]) == 18
    assert a["conclusion"] == "pipeline_check_only"
    for row in a["results"]:
        assert row["status"] == "ok"
        assert row["candidate_count"] == 1
        assert row["selected_candidate"] == 0
        assert row["test_clean_nmse"] >= 0
    for name in ["config.json", "environment.json", "provenance.json", "summary.json",
                 "stdout.log", "metrics.csv", "trials.jsonl"]:
        assert (tmp_path / "a" / name).stat().st_size > 0
    provenance = json.loads((tmp_path / "a" / "provenance.json").read_text())
    assert len(provenance["source_manifest_sha256"]) == 64
    assert provenance["selection_split"] == "validation_challenge"
    with pytest.raises(FileExistsError):
        run_experiment(config(), tmp_path / "a")


def test_failure_is_retained(tmp_path):
    cfg = config()
    cfg.update(tasks=["arithmetic"], methods=["mul"], steps=3)
    cfg["candidates"] = [{"learning_rate": 1e30, "parameter": 2.0}]
    result = run_experiment(cfg, tmp_path / "failure")
    assert result["results"][0]["status"] != "ok"
    trials = [json.loads(line) for line in (tmp_path / "failure" / "trials.jsonl").read_text().splitlines()]
    assert trials[0]["status"] != "ok"
    assert result["results"][0]["test_clean_nmse"] is None


def test_invalid_config_is_rejected_before_output(tmp_path):
    cfg = config()
    cfg["steps"] = 0
    with pytest.raises(ValueError):
        run_experiment(cfg, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_config_is_not_mutated(tmp_path):
    cfg = config()
    before = json.dumps(cfg, sort_keys=True)
    run_experiment(cfg, tmp_path / "run")
    assert json.dumps(cfg, sort_keys=True) == before


def test_normalization_baseline_has_finite_gradient_at_zero():
    import torch
    from experiments.baselines.models import InteractionModel

    model = InteractionModel("normalized_mul", 2.0, 0)
    x = torch.zeros(3, 4, requires_grad=True)
    model(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.weight.grad).all()


def test_baselines_share_actual_initial_parameters():
    import torch
    from experiments.baselines.models import METHODS, InteractionModel
    from experiments.common.data import make_split

    reference = InteractionModel("mul", 2.0, 11)
    for method in METHODS:
        model = InteractionModel(method, 2.0, 11)
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
    a = make_split("arithmetic", 10000, 8, .1, 6., 3.)
    b = make_split("arithmetic", 20000, 8, .1, 6., 3.)
    assert not torch.equal(a[0], b[0])
