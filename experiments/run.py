"""Small paired CPU screening runner. Not a confirmatory superiority experiment."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.baselines.models import METHODS, InteractionModel
from experiments.common.data import make_split

TASKS = ("corrupted_observation", "legitimate_large", "arithmetic")
ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate(config):
    for key in ("steps", "eval_every", "train_size", "validation_size", "test_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("outlier_scale", "extrapolation_scale", "projection_bound", "success_threshold"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= config["contamination"] <= 1:
        raise ValueError("contamination must be between zero and one")
    for key, allowed in (("methods", METHODS), ("tasks", TASKS)):
        if not config[key] or any(value not in allowed for value in config[key]):
            raise ValueError(f"invalid {key}")
        if len(set(config[key])) != len(config[key]):
            raise ValueError(f"duplicate {key}")
    seeds = config["seeds"]
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not config["candidates"]:
        raise ValueError("at least one candidate is required")
    for candidate in config["candidates"]:
        for key in ("learning_rate", "parameter"):
            if not math.isfinite(candidate[key]) or candidate[key] <= 0:
                raise ValueError(f"candidate {key} must be finite and positive")


def _provenance():
    files = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(p.startswith(".") or p in ("runs", "build", "dist") for p in relative.parts):
            continue
        if path.is_file() and path.suffix in (".py", ".toml", ".md", ".yml"):
            files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "remote_commit": os.environ.get("HUBERMUL_REMOTE_COMMIT"),
        "source_files": files,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "selection_split": "validation_challenge",
        "split_seed_offsets": {"train": 10000, "validation": 20000, "test": 30000},
        "dataset_generator": "experiments/common/data.py",
    }


def _split(config, task, seed, size):
    return make_split(
        task, seed, size, config["contamination"], config["outlier_scale"],
        config["extrapolation_scale"],
    )


@torch.no_grad()
def _scores(model, data, denominator):
    xc, yc, xq, yq = data
    clean = ((model(xc) - yc).square().mean() / denominator).item()
    challenge = ((model(xq) - yq).square().mean() / denominator).item()
    u, v = model.project(xq)
    coverage = ((u.abs() + v.abs()) <= model.parameter).float().mean().item()
    return clean, challenge, coverage


def _fit(config, method, candidate, seed, train, validation, denominator, emit):
    model = InteractionModel(method, candidate["parameter"], seed)
    model.constrain(config["projection_bound"])
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"])
    train_x, train_y = train
    first_threshold = None
    status = "ok"
    error = None
    clean = challenge = coverage = None
    for step in range(1, config["steps"] + 1):
        try:
            optimizer.zero_grad()
            pred = model(train_x)
            if method == "huber_loss":
                loss = F.huber_loss(pred, train_y, delta=candidate["parameter"])
            else:
                loss = F.mse_loss(pred, train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite parameter gradient")
            if method == "grad_clip":
                torch.nn.utils.clip_grad_norm_(model.parameters(), candidate["parameter"])
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite parameters")
            model.constrain(config["projection_bound"])
            if step % config["eval_every"] == 0 or step == config["steps"]:
                clean, challenge, coverage = _scores(model, validation, denominator)
                if not math.isfinite(clean) or not math.isfinite(challenge):
                    raise FloatingPointError("nonfinite validation score")
                if challenge <= config["success_threshold"] and first_threshold is None:
                    first_threshold = step
                emit(step, loss.item(), clean, challenge, norm.item(), coverage)
        except (FloatingPointError, RuntimeError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            break
    return model, {
        "status": status, "error": error, "last_step": step,
        "validation_clean_nmse": clean if status == "ok" else None,
        "validation_challenge_nmse": challenge if status == "ok" else None,
        "first_threshold_evaluated_step": first_threshold,
    }


def run_experiment(config, output_dir):
    """Run all paired trials; refuse existing directories; retain failed candidates."""
    _validate(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", config)
    _write_json(output / "provenance.json", _provenance())
    _write_json(output / "environment.json", {
        "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
        "device": "cpu", "dtype": "float32", "threads": 1,
        "cuda_available_but_not_used": torch.cuda.is_available(),
    })
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    results = []
    try:
        with (output / "metrics.csv").open("w", newline="") as mf, \
             (output / "trials.jsonl").open("w") as tf, \
             (output / "stdout.log").open("w") as log:
            writer = csv.writer(mf)
            writer.writerow(["task", "seed", "method", "candidate", "step", "train_loss",
                             "validation_clean_nmse", "validation_challenge_nmse",
                             "parameter_grad_norm", "exact_domain_fraction"])
            for task in config["tasks"]:
                for seed in config["seeds"]:
                    training = _split(config, task, seed + 10000, config["train_size"])
                    validation = _split(config, task, seed + 20000, config["validation_size"])
                    # Arithmetic learns on the clean range and tests extrapolation.
                    train = training[:2] if task == "arithmetic" else training[2:]
                    # A common normalization estimated ONLY from clean training targets.
                    denominator = training[1].square().mean().clamp_min(1e-12)
                    for method in config["methods"]:
                        best = None
                        failed = 0
                        for index, candidate in enumerate(config["candidates"]):
                            def emit(*values):
                                writer.writerow([task, seed, method, index, *values])

                            model, trial = _fit(
                                config, method, candidate, seed, train, validation,
                                denominator, emit,
                            )
                            trial.update(task=task, seed=seed, method=method, candidate=index)
                            tf.write(json.dumps(trial, allow_nan=False) + "\n")
                            tf.flush()
                            if trial["status"] == "ok":
                                score = trial["validation_challenge_nmse"]
                                if best is None or score < best[0]:
                                    best = (score, index, model, trial)
                            else:
                                failed += 1
                        row = {"task": task, "seed": seed, "method": method,
                               "candidate_count": len(config["candidates"]),
                               "failed_candidates": failed, "status": "failed",
                               "selected_candidate": None, "test_clean_nmse": None,
                               "test_challenge_nmse": None, "exact_domain_fraction": None,
                               "projection_row_norms": None,
                               "first_threshold_evaluated_step": None}
                        if best is not None:
                            _, index, model, trial = best
                            # Test data is generated/evaluated ONLY after validation selection.
                            test = _split(config, task, seed + 30000, config["test_size"])
                            clean, challenge, coverage = _scores(model, test, denominator)
                            if math.isfinite(clean) and math.isfinite(challenge):
                                row.update(status="ok", selected_candidate=index,
                                           test_clean_nmse=clean, test_challenge_nmse=challenge,
                                           exact_domain_fraction=coverage if method == "hubermul" else None,
                                           projection_row_norms=model.weight.detach().norm(dim=1).tolist(),
                                           first_threshold_evaluated_step=trial["first_threshold_evaluated_step"])
                            else:
                                row["status"] = "test_nonfinite"
                        results.append(row)
                        line = json.dumps(row, allow_nan=False)
                        log.write(line + "\n")
                        log.flush()
            summary = {"conclusion": "pipeline_check_only", "results": results}
            _write_json(output / "summary.json", summary)
    finally:
        torch.set_num_threads(previous_threads)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/smoke.toml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    summary = run_experiment(config, args.output)
    failed = sum(row["status"] != "ok" for row in summary["results"])
    print(f"Recorded {len(summary['results'])} selected runs; {failed} failed: {args.output}")
    print("Exploratory pipeline check only; no superiority conclusion.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
