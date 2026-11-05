"""Promote a validated correction model and retain the previous release."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def promote(candidate, backup_name):
    validation=json.loads((candidate/"validation-comparison.json").read_text())
    evaluation=json.loads((candidate/"evaluation.json").read_text())
    regression=json.loads((candidate/"correction-regression.json").read_text())
    metadata=json.loads((candidate/"model.json").read_text())
    manifest=json.loads((candidate/"split.json").read_text())
    expected={row["path"]:row["category"] for row in manifest["images"] if row.get("source")=="human_feedback"}
    if not validation["gate"] or evaluation["accuracy"]<0.8:
        raise ValueError("Candidate did not meet validation or assessment gates")
    actual={row.get("path",row.get("file")):row for row in regression}
    if set(actual)!=set(expected) or not all(
        actual[path]["predicted"]==category and actual[path]["modelVersion"]==metadata["modelVersion"]
        for path,category in expected.items()
    ):
        raise ValueError("Correction regressions are missing, stale, or incorrect")
    if hashlib.sha256((candidate/"mobilenetv2.keras").read_bytes()).hexdigest()!=metadata["sha256"]:
        raise ValueError("Candidate model checksum mismatch")
    if hashlib.sha256((candidate/"split.json").read_bytes()).hexdigest()!=evaluation["manifest_sha256"]:
        raise ValueError("Candidate manifest checksum mismatch")
    root=Path("artifacts")
    backup=root/backup_name
    if not backup.exists():
        backup.mkdir()
        for path in root.iterdir():
            if path.is_file():
                shutil.copy2(path,backup/path.name)
    for name in ("mobilenetv2.keras","split.json","evaluation.json","test-predictions.json",
                 "classification-report.txt","confusion-matrix.png","source-evaluation.json","training-history.json",
                 "validation-metrics.json"):
        if (candidate/name).exists():
            shutil.copy2(candidate/name,root/name)
    if (candidate/"training.log").exists():
        shutil.copy2(candidate/"training.log",root/"evaluation-run.log")
    (root/"model.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Promoted",metadata["modelVersion"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",type=Path,default=Path("experiments/feedback-metal"))
    parser.add_argument("--backup",default="before-metal-feedback")
    args=parser.parse_args()
    promote(args.candidate,args.backup)
