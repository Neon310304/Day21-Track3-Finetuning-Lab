import csv
import hashlib
import json
import math
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate


def read_json(filename):
    return json.loads((ROOT / filename).read_text(encoding="utf-8"))


def read_jsonl(filename):
    return [json.loads(line) for line in (ROOT / filename).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(path):
    content = path.read_bytes()
    return {"raw_sha256": hashlib.sha256(content).hexdigest(),
            "lf_sha256": hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()}


def score_predictions(predictions, target, regression):
    if len(predictions["target"]) != len(target):
        raise ValueError("Incomplete target predictions")
    if predictions["regression"] and len(predictions["regression"]) != len(regression):
        raise ValueError("Incomplete regression predictions")
    return {
        "target": sum(evaluate.triage_field_accuracy(prediction, row["label"]) for prediction, row in zip(predictions["target"], target)) / len(target),
        "format": sum(evaluate.has_required_keys(prediction, evaluate.TRIAGE_KEYS) for prediction in predictions["target"]) / len(target),
        "regression": sum(evaluate.keyword_recall(prediction, row["keywords"]) for prediction, row in zip(predictions["regression"], regression)) / len(regression) if predictions["regression"] else None,
    }


def main():
    target = read_jsonl("data/eval_target.jsonl")
    regression = read_jsonl("data/eval_regression.jsonl")
    baselines = read_json("results/baseline_predictions.json")
    tuned = read_json("results/evaluation_predictions.json")
    frozen = read_json("results/baselines_frozen.json")
    verdict = read_json("results/verdict.json")
    autopsy = {row["run"]: row for row in read_json("results/autopsy.json")}
    manifest = read_json("results/experiment_manifest.json")
    assertions = {}
    assertions["full_evaluation"] = len(target) == frozen["n_target"] and len(regression) == frozen["n_regression"] and not frozen["smoke_mode"]
    assertions["baseline_unchanged"] = sha256(ROOT / "results/baselines_frozen.json") == manifest["baseline_frozen_sha256"]
    assertions["original_data"] = all(sha256(ROOT / "data" / filename).startswith(expected) for filename, expected in read_json("data/checksums.json").items())
    completed = {record["stage"]: record for record in manifest["stages"] if record["exit_code"] == 0}
    assertions["core_completed"] = all(stage in completed for stage in ("nb1", "nb2", "nb3", "nb4", "nb5"))
    assertions["freeze_before_training"] = all(manifest["baseline_frozen_utc"] <= completed[stage]["started_utc"] for stage in ("nb3", "nb4"))
    filenames = {
        "nb1": "01_data_and_mask.py", "nb2": "02_baselines.py", "nb3": "03_train_correct.py",
        "nb4": "04_misconfig_autopsy.py", "nb5": "05_evaluate_and_verdict.py", "nb6": "06_merge_and_serve.py",
    }
    normalization_path = ROOT / "submission/evidence/source-normalization.json"
    if normalization_path.exists():
        normalized_sources = json.loads(normalization_path.read_text(encoding="utf-8"))
        for stage, record in completed.items():
            if stage not in normalized_sources:
                hashes = source_hashes(ROOT / "notebooks" / filenames[stage])
                if hashes["raw_sha256"] != record["script_sha256"]:
                    raise ValueError("Cannot establish LF equivalence for a new stage")
                normalized_sources[stage] = hashes
        normalization_path.write_text(json.dumps(normalized_sources, indent=2) + "\n", encoding="utf-8")
    else:
        normalized_sources = {stage: source_hashes(ROOT / "notebooks" / filenames[stage]) for stage in completed}
        if any(normalized_sources[stage]["raw_sha256"] != record["script_sha256"] for stage, record in completed.items()):
            raise ValueError("Cannot establish LF equivalence: an executed source changed")
        normalization_path.write_text(json.dumps(normalized_sources, indent=2) + "\n", encoding="utf-8")
    assertions["stage_sources_unchanged"] = all(normalized_sources[stage]["raw_sha256"] == record["script_sha256"] and source_hashes(ROOT / "notebooks" / filenames[stage])["lf_sha256"] == normalized_sources[stage]["lf_sha256"] for stage, record in completed.items())
    scores = {name: score_predictions(predictions, target, regression) for name, predictions in {**baselines, **tuned}.items()}
    for name in ("baseline_a", "baseline_b"):
        assertions[name + "_raw_matches"] = all(math.isclose(scores[name][metric], frozen[name][metric], abs_tol=1e-9) for metric in ("target", "format", "regression"))
    assertions["ft_raw_matches"] = all(round(scores["ft"][metric], 4) == verdict["comparison"][2][metric] for metric in ("target", "format", "regression"))
    for name in ("ft", "attn_only", "wrong_lr", "qlora"):
        key = "correct" if name == "ft" else name
        assertions[key + "_autopsy_matches"] = all(round(scores[name][metric], 4) == autopsy[key][metric] for metric in ("target", "format"))
    calculated = evaluate.regression_gate(evaluate.GroupScores(**scores["ft"]), evaluate.GroupScores(**scores["baseline_b"]))
    assertions["verdict_matches_raw"] = calculated.as_dict() == verdict["verdict"]
    with (ROOT / "results/runs.csv").open(encoding="utf-8") as source:
        runs = {row["run"]: row for row in csv.DictReader(source)}
    assertions["actual_equal_step_budget"] = len({int(runs[key][field]) for key in ("correct", "attn_only", "wrong_lr", "qlora") for field in ("max_steps", "actual_optimizer_steps")}) == 1
    assertions["observed_parameters_match"] = all(int(row["trainable_params"]) == int(row["observed_trainable_params"]) for row in runs.values())
    assertions["matched_parameter_budget"] = abs(int(runs["attn_only"]["trainable_params"]) - int(runs["correct"]["trainable_params"])) / int(runs["correct"]["trainable_params"]) < 0.05
    if "nb6" in completed:
        merge = read_json("results/merge_check.json")
        merge_predictions = read_json("results/merge_predictions.json")
        for key in ("before_merge", "after_merge"):
            predictions = merge_predictions[key]
            measured = sum(evaluate.triage_field_accuracy(prediction, row["label"]) for prediction, row in zip(predictions, target)) / len(target)
            assertions[key + "_raw_matches"] = len(predictions) == len(target) and math.isclose(measured, merge[key], abs_tol=1e-9)
        assertions["merge_no_regression"] = merge["n"] == len(target) and math.isclose(merge["after_merge"] - merge["before_merge"], merge["delta"], abs_tol=1e-9) and merge["delta"] >= -merge["tolerance"]
        swap = read_json("results/hot_swap.json")
        assertions["hot_swap_two_adapters"] = swap["base_loaded_once"] and len(set(swap["adapters"])) >= 2 and all(swap["predictions"].get(name) for name in swap["adapters"])
    pairs = []
    for index, row in enumerate(target):
        baseline_score = evaluate.triage_field_accuracy(baselines["baseline_b"]["target"][index], row["label"])
        tuned_score = evaluate.triage_field_accuracy(tuned["ft"]["target"][index], row["label"])
        pairs.append({"index": index, "ticket": row["input"], "label": row["label"],
                      "baseline_b_prediction": baselines["baseline_b"]["target"][index], "ft_prediction": tuned["ft"]["target"][index],
                      "baseline_b_score": baseline_score, "ft_score": tuned_score, "delta": tuned_score - baseline_score})
    artifact_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sorted((ROOT / "results").glob("*.json")) if path.name != "submission_audit.json"}
    artifact_hashes["results/runs.csv"] = sha256(ROOT / "results/runs.csv")
    artifact_hashes.update({str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sorted((ROOT / "adapters/correct").glob("adapter*")) if path.is_file()})
    payload = {"passed": all(assertions.values()), "checks": assertions, "recomputed_scores": scores,
               "qualitative_counts": {"ft_wins": sum(pair["delta"] > 0 for pair in pairs), "ft_losses": sum(pair["delta"] < 0 for pair in pairs), "ties": sum(pair["delta"] == 0 for pair in pairs)},
               "paired_examples": pairs, "artifact_sha256": artifact_hashes}
    train_inputs = {row["input"] for row in read_jsonl("data/split/train.jsonl")}
    validation_inputs = {row["input"] for row in read_jsonl("data/split/val.jsonl")}
    payload["exact_input_overlap"] = {"train_validation": len(train_inputs & validation_inputs),
                                       "train_target": sum(row["input"] in train_inputs for row in target),
                                       "validation_target": sum(row["input"] in validation_inputs for row in target)}
    regression_pairs = []
    for index, row in enumerate(regression):
        baseline_score = evaluate.keyword_recall(baselines["baseline_b"]["regression"][index], row["keywords"])
        tuned_score = evaluate.keyword_recall(tuned["ft"]["regression"][index], row["keywords"])
        regression_pairs.append({"index": index, "instruction": row["instruction"], "keywords": row["keywords"],
                                 "baseline_b_prediction": baselines["baseline_b"]["regression"][index], "ft_prediction": tuned["ft"]["regression"][index],
                                 "baseline_b_score": baseline_score, "ft_score": tuned_score, "delta": tuned_score - baseline_score})
    payload["paired_regression_examples"] = regression_pairs
    (ROOT / "results/submission_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, passed in assertions.items():
        print(f"{'OK' if passed else 'FAIL'} {name}")
    print(payload["qualitative_counts"])
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
