import importlib.util
from pathlib import Path

import pytest


def auditor():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_submission.py"
    spec = importlib.util.spec_from_file_location("submission_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_recomputes_scores_from_outputs():
    module = auditor()
    label = {"intent": "hoi_thong_tin", "urgency": "thap", "product": "ba lô", "sentiment": "trung_tinh"}
    predictions = {"target": ['{"intent":"hoi_thong_tin","urgency":"thap","product":"ba lo","sentiment":"trung_tinh"}'], "regression": ["Hà Nội"]}
    assert module.score_predictions(predictions, [{"label": label}], [{"keywords": ["Hà Nội"]}]) == {"target": 1.0, "format": 1.0, "regression": 1.0}


def test_audit_rejects_incomplete_predictions_and_marks_unmeasured_regression():
    module = auditor()
    with pytest.raises(ValueError):
        module.score_predictions({"target": [], "regression": []}, [{"label": {}}], [])
    with pytest.raises(ValueError):
        module.score_predictions({"target": ["{}"], "regression": ["one"]}, [{"label": {}}], [{"keywords": []}, {"keywords": []}])
    scores = module.score_predictions({"target": ["{}"], "regression": []}, [{"label": {}}], [{"keywords": []}])
    assert scores["regression"] is None


def test_source_hash_tolerates_only_crlf_conversion(tmp_path):
    module = auditor()
    path = tmp_path / "stage.py"
    path.write_bytes(b"print('measured')\r\n")
    windows = module.source_hashes(path)
    path.write_bytes(b"print('measured')\n")
    linux = module.source_hashes(path)
    assert windows["raw_sha256"] != linux["raw_sha256"]
    assert windows["lf_sha256"] == linux["lf_sha256"]
    path.write_bytes(b"print('changed')\n")
    assert module.source_hashes(path)["lf_sha256"] != windows["lf_sha256"]
