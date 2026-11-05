"""Compare frozen candidate and baseline on validation and test by source."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import image_array

# Historical comparison for the three-class experiment.
CATEGORIES = ("ORGANIC", "PLASTIC", "PAPER")

def compare():
    directory = Path("experiments/taco")
    manifest = json.loads((directory / "split.json").read_text())
    rows = [row for row in manifest["images"] if row["split"] in {"validation", "test"}]
    arrays = np.stack([image_array(Path("../temp-image") / row["path"]) for row in rows])
    labels = np.array([CATEGORIES.index(row["category"]) for row in rows])
    result = {"models": {}, "manifest_sha256": hashlib.sha256((directory / "split.json").read_bytes()).hexdigest()}
    for name, path in (("baseline", Path("artifacts/baseline-v1/mobilenetv2.keras")),
                       ("candidate", directory / "mobilenetv2.keras")):
        model = keras.models.load_model(path, compile=False)
        predicted = np.concatenate([model(arrays[start:start+32], training=False).numpy().argmax(axis=1)
                                    for start in range(0, len(rows), 32)])
        metrics = {}
        for split in ("validation", "test"):
            for source in ("all", "kaggle", "taco"):
                selected = [i for i, row in enumerate(rows)
                            if row["split"] == split and (source == "all" or row["source"] == source)]
                metrics[f"{split}_{source}"] = {
                    "images": len(selected),
                    "accuracy": float(accuracy_score(labels[selected], predicted[selected])),
                    "classification_report": classification_report(labels[selected], predicted[selected],
                        labels=list(range(3)), target_names=CATEGORIES, output_dict=True, zero_division=0),
                }
        result["models"][name] = metrics
    (directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, metrics in result["models"].items():
        print(name, {key: round(value["accuracy"], 4) for key, value in metrics.items()}, flush=True)

if __name__ == "__main__":
    compare()
