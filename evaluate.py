"""Evaluate an exported model on the held-out test images."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

from preprocessing import CATEGORIES, image_array, load_rgb
from bottle_detector import BottleDetector, MODEL_PATH
from material_inference import prediction


def evaluate(root, manifest_path, output, model_dir=None):
    manifest = json.loads(manifest_path.read_text())
    groups = {}
    for row in manifest["images"]:
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("Image group appears in multiple splits")
        if hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Dataset content changed: {row['path']}")
    rows = [row for row in manifest["images"] if row["split"] == "test"]
    output.mkdir(parents=True, exist_ok=True)
    model_path = (model_dir or output) / "mobilenetv2.keras"
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path = model_path.parent / "model.json"
    source_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if source_metadata.get("sha256") != model_sha:
        source_metadata = {}
    categories = tuple(source_metadata.get("classes", CATEGORIES))
    detector = None
    inference = source_metadata.get("inference")
    if inference:
        if inference.get("policy") != "bottle_consensus_v1":
            raise ValueError("Unsupported inference policy")
        if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != inference["detectorSha256"]:
            raise ValueError("Detector checksum mismatch")
        detector = BottleDetector()
    model = keras.models.load_model(model_path, compile=False)
    image_size = model.input_shape[1]
    if image_size not in (160, 224):
        raise ValueError("Unsupported model image size")
    probabilities = []
    for offset in range(0, len(rows), 32):
        batch = np.stack([image_array(root / row["path"], image_size) for row in rows[offset:offset + 32]])
        baseline = model(batch, training=False).numpy()
        for row, scores in zip(rows[offset:offset + 32], baseline):
            if detector:
                scores, _ = prediction(load_rgb(root / row["path"]), model, categories,
                                       image_size, detector, baseline=scores)
            probabilities.append(scores)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    labels = np.array([categories.index(row["category"]) for row in rows])
    matrix = confusion_matrix(labels, predicted, labels=range(len(categories)))
    accuracy = float(accuracy_score(labels, predicted))
    report = classification_report(labels, predicted, labels=range(len(categories)),
                                   target_names=categories, output_dict=True, zero_division=0)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {"modelVersion": f"mobilenetv2-{model_sha[:12]}", "classes": list(categories),
                "imageSize": image_size, "sha256": model_sha}
    if inference:
        metadata["modelVersion"] = source_metadata["modelVersion"]
        metadata["inference"] = inference
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "evaluation.json").write_text(json.dumps({
        "accuracy": accuracy, "target_met": accuracy >= 0.80, "test_images": len(rows),
        "labels": list(categories), "counts": manifest["counts"],
        "confusion_matrix": matrix.tolist(), "classification_report": report,
        "seed": manifest["seed"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_sha256": model_sha,
        "pipeline_version": metadata["modelVersion"],
        "inference_policy": inference["policy"] if inference else "whole_image",
        "detector_sha256": inference["detectorSha256"] if inference else None,
    }, indent=2) + "\n")
    domains = {}
    for domain in sorted({row.get("source", "kaggle") for row in rows}):
        selected = [i for i, row in enumerate(rows) if row.get("source", "kaggle") == domain]
        domains[domain] = {
            "images": len(selected),
            "accuracy": float(accuracy_score(labels[selected], predicted[selected])),
            "classification_report": classification_report(labels[selected], predicted[selected],
                labels=range(len(categories)), target_names=categories, output_dict=True, zero_division=0),
        }
    (output / "source-evaluation.json").write_text(json.dumps(domains, indent=2) + "\n")
    (output / "test-predictions.json").write_text(json.dumps([
        {"path": row["path"], "actual": row["category"], "predicted": categories[int(label)],
         "probabilities": scores.tolist()}
        for row, label, scores in zip(rows, predicted, probabilities)
    ], indent=2) + "\n")
    text_report = classification_report(labels, predicted, labels=range(len(categories)),
                                        target_names=categories, digits=4, zero_division=0)
    (output / "classification-report.txt").write_text(text_report)
    fig, ax = plt.subplots(figsize=(max(6.2, len(categories)*0.9), max(5.1, len(categories)*0.8)), layout="constrained")
    ConfusionMatrixDisplay(matrix, display_labels=categories).plot(ax=ax, colorbar=False, cmap="Greens")
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(f"Test set • {len(rows)} images • accuracy {accuracy:.1%}", pad=12)
    fig.savefig(output / "confusion-matrix.png", dpi=180)
    plt.close(fig)
    print(text_report)
    print(f"Test accuracy: {accuracy:.2%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../temp-image"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/split.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.manifest, args.output, args.model_dir)
