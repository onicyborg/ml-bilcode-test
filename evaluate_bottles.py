"""Evaluate a fixed bottle-consensus policy against the deployed classifier."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "3")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from bottle_detector import BottleDetector, MODEL_PATH
from material_inference import prediction
from preprocessing import CATEGORIES, load_rgb, array_from_image

OUT = Path("experiments/bottle-consensus")
ROOT = Path("../temp-image")


def metrics(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "classification_report": classification_report(actual, predicted, labels=range(len(CATEGORIES)),
            target_names=CATEGORIES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=range(len(CATEGORIES))).tolist(),
        "glass_as_plastic": int(((actual==CATEGORIES.index("GLASS")) & (predicted==CATEGORIES.index("PLASTIC"))).sum()),
    }


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("artifacts/split.json").read_text())
    metadata = json.loads(Path("artifacts/model.json").read_text())
    model = keras.models.load_model("artifacts/mobilenetv2.keras", compile=False)
    detector = BottleDetector()
    results = {}
    for split in ("validation", "test"):
        rows = [row for row in manifest["images"] if row["split"]==split]
        baseline = []
        for offset in range(0, len(rows), 32):
            arrays = np.stack([array_from_image(load_rgb(ROOT/row["path"]), metadata["imageSize"])
                               for row in rows[offset:offset+32]])
            baseline.extend(model(arrays, training=False).numpy())
        proposed = []
        actual = np.array([CATEGORIES.index(row["category"]) for row in rows])
        changes = []
        for index, (row, scores) in enumerate(zip(rows, baseline)):
            if index % 250 == 0:
                print(f"{split}: {index}/{len(rows)}", flush=True)
            output, used = prediction(load_rgb(ROOT/row["path"]), model, CATEGORIES, metadata["imageSize"],
                                      detector, baseline=scores)
            proposed.append(output)
            if used:
                changes.append({"path": row["path"], "actual": row["category"],
                    "before": CATEGORIES[int(np.argmax(scores))], "after": CATEGORIES[int(np.argmax(output))]})
        baseline, proposed = np.asarray(baseline), np.asarray(proposed)
        before, after = baseline.argmax(axis=1), proposed.argmax(axis=1)
        results[split] = {"baseline": metrics(actual, before), "candidate": metrics(actual, after), "routed": changes}
        (OUT/f"{split}-predictions.json").write_text(json.dumps([
            {"path": row["path"], "actual": row["category"], "predicted": CATEGORIES[int(p.argmax())],
             "probabilities": p.tolist()} for row,p in zip(rows,proposed)], indent=2)+"\n")
        if split=="validation":
            b, c = results[split]["baseline"],results[split]["candidate"]
            gate = (c["balanced_accuracy"] >= b["balanced_accuracy"]-0.005
                    and c["glass_as_plastic"] <= b["glass_as_plastic"]
                    and c["classification_report"]["PLASTIC"]["f1-score"] >= b["classification_report"]["PLASTIC"]["f1-score"]-0.01)
            results["validation_gate"] = bool(gate)
            print("Validation gate",gate, "routed",len(changes), flush=True)
            if not gate:
                break
    results["detector_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    results["model_sha256"] = metadata["sha256"]
    (OUT/"comparison.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({s: {name: results[s][name]["accuracy"] for name in ("baseline","candidate")}
                      for s in ("validation","test") if s in results}), flush=True)


if __name__ == "__main__":
    evaluate()
