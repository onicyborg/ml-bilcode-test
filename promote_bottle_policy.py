"""Promote a validated bottle policy and retain the whole-image release."""

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from preprocessing import CATEGORIES


def promote():
    candidate = Path("experiments/bottle-consensus")
    comparison = json.loads((candidate / "comparison.json").read_text())
    field = json.loads((candidate / "field-example.json").read_text())
    assert comparison["validation_gate"] and comparison["test"]["candidate"]["accuracy"] >= 0.8
    assert field["correct"] and field["predicted"] == "GLASS"
    root = Path("artifacts")
    backup = root / "whole-image-10class"
    if not backup.exists():
        backup.mkdir()
        for path in root.iterdir():
            if path.is_file():
                shutil.copy2(path, backup / path.name)
    metadata = json.loads((candidate / "model.json").read_text())
    assert hashlib.sha256((root / "mobilenetv2.keras").read_bytes()).hexdigest() == metadata["sha256"]
    manifest = json.loads((root / "split.json").read_text())
    evaluation = json.loads((root / "evaluation.json").read_text())
    measured = comparison["test"]["candidate"]
    for key in ("accuracy", "classification_report", "confusion_matrix"):
        evaluation[key] = measured[key]
    evaluation["pipeline_version"] = metadata["modelVersion"]
    evaluation["detector_sha256"] = metadata["inference"]["detectorSha256"]
    evaluation["inference_policy"] = "bottle_consensus_v1"
    (root / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    predictions = json.loads((candidate / "test-predictions.json").read_text())
    report = measured["classification_report"]
    lines = ["Category       Precision    Recall        F1    Support"]
    for category in CATEGORIES:
        r = report[category]
        lines.append(f"{category:12} {r['precision']:10.4f} {r['recall']:9.4f} {r['f1-score']:9.4f} {r['support']:8.0f}")
    (root / "classification-report.txt").write_text("\n".join(lines) + "\n")
    by_path = {row["path"]: row for row in manifest["images"]}
    source_metrics = {}
    for source in sorted({by_path[row["path"]].get("source", "kaggle") for row in predictions}):
        subset = [row for row in predictions if by_path[row["path"]].get("source", "kaggle") == source]
        actual = [row["actual"] for row in subset]
        predicted = [row["predicted"] for row in subset]
        source_metrics[source] = {
            "images": len(subset), "accuracy": float(accuracy_score(actual, predicted)),
            "classification_report": classification_report(actual, predicted, labels=list(CATEGORIES),
                target_names=CATEGORIES, output_dict=True, zero_division=0),
        }
    (root / "source-evaluation.json").write_text(json.dumps(source_metrics, indent=2) + "\n")
    shutil.copy2(candidate / "test-predictions.json", root / "test-predictions.json")
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    ConfusionMatrixDisplay(np.array(measured["confusion_matrix"]), display_labels=CATEGORIES).plot(ax=ax, colorbar=False, cmap="Greens")
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(f"Bottle policy • {len(predictions)} test samples • accuracy {measured['accuracy']:.1%}")
    fig.savefig(root / "confusion-matrix.png", dpi=180)
    plt.close(fig)
    # Publish the policy metadata after the matching evaluation artifacts.
    (root / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Promoted", metadata["modelVersion"])


if __name__ == "__main__":
    promote()
