"""Train MobileNetV2 transfer learning and evaluate a held-out test split."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from preprocessing import CATEGORIES, IMAGE_SIZE, image_array


def embed(extractor, rows, root, *, flip=False):
    features, labels = [], []
    for start in range(0, len(rows), 32):
        chunk = rows[start:start + 32]
        images = np.stack([image_array(root / row["path"]) for row in chunk])
        features.append(extractor(images, training=False).numpy())
        labels.extend(CATEGORIES.index(row["category"]) for row in chunk)
        if flip:
            features.append(extractor(images[:, :, ::-1].copy(), training=False).numpy())
            labels.extend(CATEGORIES.index(row["category"]) for row in chunk)
    return np.concatenate(features), np.array(labels)


def train(root, manifest_path, output, fine_tune_epochs=0):
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["images"]:
        if hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Dataset content changed: {row['path']}")
    keras.utils.set_random_seed(manifest["seed"])
    groups = {}
    for row in manifest["images"]:
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("Image group appears in multiple splits")

    splits = {name: [row for row in manifest["images"] if row["split"] == name]
              for name in ("train", "validation", "test")}
    expected = set(CATEGORIES)
    for name, rows in splits.items():
        if {row["category"] for row in rows} != expected:
            raise ValueError(f"Every category must be represented in {name}")

    base = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False,
        weights="imagenet", pooling="avg",
    )
    base.trainable = False
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype="uint8")
    scaled = keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    features = base(scaled, training=False)
    extractor = keras.Model(inputs, features, name="imagenet_features")

    print("Extracting features for train and validation", flush=True)
    train_x, train_y = embed(extractor, splits["train"], root, flip=True)
    val_x, val_y = embed(extractor, splits["validation"], root)
    head = keras.Sequential([
        keras.Input((base.output_shape[-1],)),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(96, activation="relu", kernel_regularizer=keras.regularizers.L2(1e-4)),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(len(CATEGORIES), activation="softmax"),
    ], name="waste_classifier")
    head.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    training = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(
        len(train_y), seed=manifest["seed"]).batch(32).with_options(options)
    validation = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(32).with_options(options)
    class_counts = np.bincount(train_y, minlength=len(CATEGORIES))
    class_weights = {index: float(len(train_y) / (len(CATEGORIES) * count))
                     for index, count in enumerate(class_counts)}
    history = head.fit(training, class_weight=class_weights, validation_data=validation, epochs=25, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ])

    model = keras.Model(inputs, head(features), name="siklus_mobilenetv2")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "mobilenetv2.keras"
    training_record = {
        "history": history.history,
        "class_weights": class_weights,
        "selected_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "classes": list(CATEGORIES),
    }
    model.save(model_path)
    if fine_tune_epochs:
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal", seed=manifest["seed"]),
            keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=manifest["seed"] + 1),
            keras.layers.RandomContrast(0.1, seed=manifest["seed"] + 2),
        ])

        def pixels(rows, augment=False):
            paths = [str(root / row["path"]) for row in rows]
            labels = [CATEGORIES.index(row["category"]) for row in rows]
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
            if augment:
                dataset = dataset.shuffle(len(rows), seed=manifest["seed"])
            def decode(path, label):
                image = tf.numpy_function(lambda value: image_array(os.fsdecode(value)), [path], tf.uint8)
                image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
                return image, label
            dataset = dataset.map(decode, num_parallel_calls=2).batch(16)
            if augment:
                dataset = dataset.map(lambda image, label: (
                    tf.cast(tf.clip_by_value(augmentation(image, training=True), 0, 255), tf.uint8), label),
                    num_parallel_calls=1)
            return dataset.with_options(options).prefetch(1)

        validation_pixels = pixels(splits["validation"])
        base.trainable = True
        for index, layer in enumerate(base.layers):
            layer.trainable = index >= len(base.layers) - 30 and not isinstance(layer, keras.layers.BatchNormalization)
        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        baseline_loss = model.evaluate(validation_pixels, verbose=0)[0]
        print("Fine-tuning last 30 backbone layers, keeping BatchNormalization frozen", flush=True)
        fine_history = model.fit(pixels(splits["train"], augment=True),
            validation_data=validation_pixels, epochs=fine_tune_epochs, class_weight=class_weights,
            verbose=2, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
        chosen = min(fine_history.history["val_loss"]) < baseline_loss
        if chosen:
            model.save(model_path)
        training_record["fine_tuning"] = {
            "history": fine_history.history, "baseline_validation_loss": baseline_loss,
            "selected": chosen,
            "best_epoch": int(np.argmin(fine_history.history["val_loss"])) + 1,
        }

    (output / "training-history.json").write_text(json.dumps(training_record, indent=2) + "\n")
    from evaluate import evaluate
    evaluate(root, manifest_path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../temp-image"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/split.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    args = parser.parse_args()
    train(args.data, args.manifest, args.output, args.fine_tune_epochs)
