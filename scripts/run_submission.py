import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import device
from labkit.config import NAIVE_PROMPT, OPTIMIZED_PROMPT, get_tier, training_epochs


STAGES = {
    "nb1": "notebooks/01_data_and_mask.py",
    "nb2": "notebooks/02_baselines.py",
    "nb3": "notebooks/03_train_correct.py",
    "nb4": "notebooks/04_misconfig_autopsy.py",
    "nb5": "notebooks/05_evaluate_and_verdict.py",
    "nb6": "notebooks/06_merge_and_serve.py",
}


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measured submission run with logs and baseline-before-training evidence")
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    args = parser.parse_args()
    if os.environ.get("EVAL_LIMIT", "0") not in {"", "0"}:
        parser.error("Submission runner requires the full evaluation sets")
    results = ROOT / "results"
    evidence = ROOT / "submission" / "evidence"
    results.mkdir(exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    manifest_path = results / "experiment_manifest.json"
    tier = get_tier()
    settings = {"tier": tier.name, "model": tier.model_id, "max_length": tier.max_length,
                "per_device_batch": tier.per_device_batch, "grad_accum": tier.grad_accum,
                "epochs": training_epochs(), "mask_mode": os.environ.get("MASK_MODE", "assistant-only"),
                "naive_prompt_sha256": hashlib.sha256(NAIVE_PROMPT.encode("utf-8")).hexdigest(),
                "optimized_prompt_sha256": hashlib.sha256(OPTIMIZED_PROMPT.encode("utf-8")).hexdigest()}
    sources = {name: digest(ROOT / "data" / name) for name in ("train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl")}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else \
        {"started_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
         "source_sha256": sources, "device": device.describe(), "python": sys.version, "stages": []}
    if manifest["settings"] != settings or manifest["source_sha256"] != sources:
        parser.error("Experiment configuration/data changed; use a separate clean experiment, not these artifacts")
    for stage in args.stages:
        if any(record["stage"] == stage and record["exit_code"] == 0 for record in manifest["stages"]):
            parser.error(f"{stage} already completed; refusing to overwrite measured evidence")
        if stage in {"nb3", "nb4", "nb5", "nb6"}:
            if not manifest.get("baseline_frozen_sha256"):
                parser.error("NB2 must complete and freeze the baseline before training/evaluation")
            if digest(results / "baselines_frozen.json") != manifest["baseline_frozen_sha256"]:
                parser.error("Frozen baseline artifact changed")
            if stage in {"nb3", "nb4"} and any(record["stage"] in {"nb5", "nb6"} for record in manifest["stages"]):
                parser.error("Refusing to tune training after inspecting evaluation")
        record = {"stage": stage, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "script_sha256": digest(ROOT / STAGES[stage])}
        started = time.perf_counter()
        filename = stage + "-" + str(len(manifest["stages"]) + 1) + ".log"
        print(f"Starting {stage}, output in submission/evidence/{filename}", flush=True)
        with (evidence / filename).open("x", encoding="utf-8") as destination:
            process = subprocess.Popen([sys.executable, "-u", STAGES[stage]], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace",
                                       env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
            try:
                for line in process.stdout:
                    destination.write(line)
                    destination.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        record.update({"finished_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code,
                       "seconds": time.perf_counter() - started, "log": "submission/evidence/" + filename})
        manifest["stages"].append(record)
        if stage == "nb2" and code == 0:
            manifest["baseline_frozen_sha256"] = digest(results / "baselines_frozen.json")
            manifest["baseline_frozen_utc"] = record["finished_utc"]
        save(manifest_path, manifest)
        if code:
            print(f"{stage} failed; evidence preserved, no downstream stage started", flush=True)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
