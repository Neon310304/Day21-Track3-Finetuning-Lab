import importlib.util
import json
import sys
from pathlib import Path

import pytest


def runner(tmp_path, monkeypatch):
    location = Path(__file__).resolve().parents[1] / "scripts" / "run_submission.py"
    spec = importlib.util.spec_from_file_location("submission_runner", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.device, "describe", lambda: {"device": "cpu", "name": "unit fixture"})
    monkeypatch.delenv("EVAL_LIMIT", raising=False)
    directory = tmp_path / "data"
    directory.mkdir()
    for filename in ("train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl"):
        (directory / filename).write_text('{}\n', encoding="utf-8")
    for stage, filename in module.STAGES.items():
        path = tmp_path / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text("from pathlib import Path\nprint('unit fixture stage')\n"
                        + ("Path('results/baselines_frozen.json').write_text('{}')\n" if stage == "nb2" else ""),
                        encoding="utf-8")
    return module


def test_runner_logs_freeze_before_training_and_refuses_overwrite(tmp_path, monkeypatch):
    module = runner(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_submission.py", "nb1", "nb2", "nb3"])
    assert module.main() == 0
    manifest = json.loads((tmp_path / "results" / "experiment_manifest.json").read_text())
    assert [record["stage"] for record in manifest["stages"]] == ["nb1", "nb2", "nb3"]
    assert manifest["baseline_frozen_utc"] <= manifest["stages"][2]["started_utc"]
    assert all((tmp_path / record["log"]).is_file() for record in manifest["stages"])
    assert manifest["baseline_frozen_sha256"] == module.digest(tmp_path / "results" / "baselines_frozen.json")
    with pytest.raises(SystemExit):
        module.main()


def test_runner_rejects_training_without_baseline_and_smoke_submission(tmp_path, monkeypatch):
    module = runner(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_submission.py", "nb3"])
    with pytest.raises(SystemExit):
        module.main()
    monkeypatch.setenv("EVAL_LIMIT", "8")
    monkeypatch.setattr(sys, "argv", ["run_submission.py", "nb1"])
    with pytest.raises(SystemExit):
        module.main()


def test_runner_rejects_changed_data_or_frozen_baseline(tmp_path, monkeypatch):
    module = runner(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_submission.py", "nb2"])
    assert module.main() == 0
    baseline = tmp_path / "results" / "baselines_frozen.json"
    baseline.write_text('{"changed":true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_submission.py", "nb3"])
    with pytest.raises(SystemExit):
        module.main()
    (tmp_path / "data" / "train_seed.jsonl").write_text('{"changed":true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_submission.py", "nb1"])
    with pytest.raises(SystemExit):
        module.main()


def test_runner_keeps_failed_stage_and_stops_downstream(tmp_path, monkeypatch):
    module = runner(tmp_path, monkeypatch)
    (tmp_path / module.STAGES["nb1"]).write_text("raise SystemExit(7)\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_submission.py", "nb1", "nb2"])
    assert module.main() == 7
    manifest = json.loads((tmp_path / "results" / "experiment_manifest.json").read_text())
    assert len(manifest["stages"]) == 1 and manifest["stages"][0]["exit_code"] == 7
    assert not (tmp_path / "results" / "baselines_frozen.json").exists()
