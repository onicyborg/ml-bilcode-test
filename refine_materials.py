"""Fine-tune at 224px using source-grouped context and material-focused rehearsal."""

import hashlib
import json
import os
from pathlib import Path
import random
from collections import Counter, defaultdict

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras

from preprocessing import CATEGORIES, image_array
from prepare_multiclass import MAPPING

DIRECTORY = Path("experiments/materials224")
ROOT = Path("../temp-image")
IMAGE_SIZE = 224
SEED = 84
FOCUS = {"GLASS", "PLASTIC", "METAL"}


def prepare():
    manifest = json.loads(Path("artifacts/split.json").read_text())
    rows = [dict(row) for row in manifest["images"]]
    source_groups = {}
    for row in rows:
        if row["source"] == "taco":
            key = row["source_image"]
            value = (row["split"], row["group"])
            if source_groups.setdefault(key, value) != value:
                raise ValueError("TACO source photo appears in multiple splits")
    annotations = json.loads((ROOT / "TACO/TACO/data/annotations.json").read_text())
    category_names = {row["id"]: row["name"] for row in annotations["categories"]}
    per_image = defaultdict(list)
    for annotation in annotations["annotations"]:
        per_image[annotation["image_id"]].append(annotation)
    for photo in annotations["images"]:
        if photo["file_name"] not in source_groups:
            continue
        objects = per_image[photo["id"]]
        categories = {MAPPING.get(category_names[obj["category_id"]]) for obj in objects}
        if len(categories) != 1 or next(iter(categories)) not in FOCUS:
            continue
        # Keep scenes with enough annotated material to support a whole-image label.
        if sum(obj["area"] for obj in objects) / (photo["width"] * photo["height"]) < 0.05:
            continue
        path = "TACO/TACO/data/" + photo["file_name"]
        with Image.open(ROOT / path) as image:
            if image.width * image.height > 25_000_000:
                continue
        split, group = source_groups[photo["file_name"]]
        rows.append({
            "path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            "category": next(iter(categories)), "split": split, "group": group,
            "source": "taco-context", "source_image": photo["file_name"],
        })
    groups = {}
    for row in rows:
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("Image group crosses dataset splits")
    manifest["images"] = rows
    manifest["counts"] = {
        split: dict(Counter(row["category"] for row in rows if row["split"] == split))
        for split in ("train", "validation", "test")
    }
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    (DIRECTORY / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def pixels(rows, size, augment=False):
    paths = [str(ROOT / row["path"]) for row in rows]
    labels = [CATEGORIES.index(row["category"]) for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if augment:
        dataset = dataset.shuffle(len(rows), seed=SEED)
    def decode(path, label):
        image = tf.numpy_function(lambda value: image_array(os.fsdecode(value), size), [path], tf.uint8)
        image.set_shape((size, size, 3))
        return image, label
    dataset = dataset.map(decode, num_parallel_calls=2).batch(16)
    if augment:
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal", seed=SEED),
            keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=SEED+1),
            keras.layers.RandomZoom(0.12, fill_mode="reflect", seed=SEED+2),
            keras.layers.RandomContrast(0.12, seed=SEED+3),
        ])
        dataset = dataset.map(lambda image, label: (
            tf.cast(tf.clip_by_value(augmentation(image, training=True), 0, 255), tf.uint8), label),
            num_parallel_calls=1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return dataset.with_options(options).prefetch(1)


def validation_metrics(model, rows):
    probabilities = np.concatenate([model(images, training=False).numpy()
                                    for images, _ in pixels(rows, model.input_shape[1])])
    actual = np.array([CATEGORIES.index(row["category"]) for row in rows])
    predicted = probabilities.argmax(axis=1)
    from sklearn.metrics import classification_report, balanced_accuracy_score
    return {
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "classification_report": classification_report(actual, predicted, labels=range(len(CATEGORIES)),
            target_names=CATEGORIES, output_dict=True, zero_division=0),
        "glass_as_plastic": int(((actual == CATEGORIES.index("GLASS")) &
                                (predicted == CATEGORIES.index("PLASTIC"))).sum()),
    }


def refine():
    manifest = prepare()
    for row in manifest["images"]:
        if hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Source image changed")
    by_category = defaultdict(list)
    for row in manifest["images"]:
        if row["split"] == "train":
            by_category[row["category"]].append(row)
    rng = random.Random(SEED)
    training_rows = []
    for category in CATEGORIES:
        rows = by_category[category]
        rng.shuffle(rows)
        training_rows.extend(rows if category in FOCUS else rows[:250])
    validation_rows = [row for row in manifest["images"] if row["split"] == "validation"]
    print("Training subset:", dict(Counter(row["category"] for row in training_rows)), flush=True)
    print("Added context photos:", dict(Counter(row["split"] for row in manifest["images"] if row["source"] == "taco-context")), flush=True)
    keras.utils.set_random_seed(SEED)
    old = keras.models.load_model("artifacts/mobilenetv2.keras", compile=False)
    print("Measuring baseline on fixed validation", flush=True)
    baseline = validation_metrics(old, validation_rows)
    old_base = next(layer for layer in old.layers if isinstance(layer, keras.Model) and "mobilenet" in layer.name)
    old_head = old.get_layer("waste_classifier")
    base = keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False, weights=None, pooling="avg")
    base.set_weights(old_base.get_weights())
    head = keras.models.clone_model(old_head)
    head.set_weights(old_head.get_weights())
    base.trainable = True
    for index, layer in enumerate(base.layers):
        layer.trainable = index >= len(base.layers)-45 and not isinstance(layer, keras.layers.BatchNormalization)
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype="uint8")
    features = base(keras.layers.Rescaling(1/127.5, offset=-1)(inputs), training=False)
    model = keras.Model(inputs, head(features), name="siklus_materials224")
    model.compile(optimizer=keras.optimizers.Adam(8e-6), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    counts = Counter(row["category"] for row in training_rows)
    class_weights = {index: len(training_rows)/(len(CATEGORIES)*counts[category]) *
                     (1.4 if category=="GLASS" else 1.15 if category=="PLASTIC" else 1)
                     for index, category in enumerate(CATEGORIES)}
    history = model.fit(pixels(training_rows, IMAGE_SIZE, True),
        validation_data=pixels(validation_rows, IMAGE_SIZE), epochs=6, class_weight=class_weights,
        verbose=2, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
    model.save(DIRECTORY / "mobilenetv2.keras")
    candidate = validation_metrics(model, validation_rows)
    gate = (
        candidate["balanced_accuracy"] >= baseline["balanced_accuracy"]-0.01
        and candidate["glass_as_plastic"] < baseline["glass_as_plastic"]
        and candidate["classification_report"]["PLASTIC"]["f1-score"] >=
            baseline["classification_report"]["PLASTIC"]["f1-score"]-0.02
    )
    record = {
        "history": history.history, "best_epoch": int(np.argmin(history.history["val_loss"]))+1,
        "baseline_validation": baseline, "candidate_validation": candidate, "validation_gate": bool(gate),
        "seed": SEED, "training_files": [row["path"] for row in training_rows],
        "training_subset_counts": dict(counts), "class_weights": class_weights,
        "initial_model_sha256": hashlib.sha256(Path("artifacts/mobilenetv2.keras").read_bytes()).hexdigest(),
    }
    (DIRECTORY / "training-history.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Validation gate:", gate, "GLASS→PLASTIC", baseline["glass_as_plastic"], "→", candidate["glass_as_plastic"], flush=True)
    from evaluate import evaluate
    evaluate(ROOT, DIRECTORY / "split.json", DIRECTORY)


if __name__ == "__main__":
    refine()
