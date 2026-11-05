"""Train the material classifier with explicit, train-only human corrections."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score, classification_report

from dataset import fingerprint
from preprocessing import CATEGORIES, image_array, load_rgb, array_from_image
from bottle_detector import BottleDetector
from material_inference import prediction

ROOT = Path("../temp-image")
OUT = Path("experiments/feedback-metal")
SEED = 125


def feedback_images(image, count):
    rng = random.Random(SEED)
    images = [array_from_image(image)]
    for _ in range(count-1):
        zoom = rng.uniform(0, 0.08)
        left, top = round(image.width*zoom), round(image.height*zoom)
        sample = image.crop((left, top, image.width-left, image.height-top))
        sample = sample.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR,
                               expand=False, fillcolor=(127,127,127))
        sample = ImageEnhance.Brightness(sample).enhance(rng.uniform(0.85,1.15))
        sample = ImageEnhance.Contrast(sample).enhance(rng.uniform(0.85,1.15))
        if rng.random()<0.5:
            sample=ImageOps.mirror(sample)
        images.append(array_from_image(sample))
    return images


def evaluate_validation(model, rows, detector):
    scores = []
    # Use the deployed policy for comparison, not only the classifier head.
    for offset in range(0, len(rows), 32):
        chunk = rows[offset:offset+32]
        batch = np.stack([image_array(ROOT/row["path"]) for row in chunk])
        baseline = model(batch, training=False).numpy()
        for row, initial in zip(chunk,baseline):
            output,_ = prediction(load_rgb(ROOT/row["path"]),model,CATEGORIES,160,detector,baseline=initial)
            scores.append(output)
    labels=np.array([CATEGORIES.index(row["category"]) for row in rows])
    predicted=np.asarray(scores).argmax(axis=1)
    return {
        "balanced_accuracy":float(balanced_accuracy_score(labels,predicted)),
        "classification_report":classification_report(labels,predicted,labels=range(len(CATEGORIES)),
            target_names=CATEGORIES,output_dict=True,zero_division=0),
    }


def train(initial_dir):
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((initial_dir/"split.json").read_text())
    manifest["images"]=[row for row in manifest["images"] if row.get("source")!="human_feedback"]
    manifest["counts"]={split:dict(Counter(row["category"] for row in manifest["images"] if row["split"]==split))
                        for split in ("train","validation","test")}
    for row in manifest["images"]:
        if hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError(f"Dataset content changed: {row['path']}")
    feedback=json.loads(Path("feedback.json").read_text())["examples"]
    feedback_hashes={hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest() for row in feedback}
    feedback_fingerprints=[fingerprint(load_rgb(ROOT/row["path"])) for row in feedback]
    for row in manifest["images"]:
        if row["split"]=="train":
            continue
        if row["sha256"] in feedback_hashes:
            raise ValueError("Feedback image already appears in a held-out split")
        h,v=fingerprint(load_rgb(ROOT/row["path"]))
        if any((h-hh==0 or (h^hh).bit_count()<=4) and (v^vv).bit_count()<=4
               for hh,vv in feedback_fingerprints):
            raise ValueError(f"Potential feedback leakage into held-out data: {row['path']}")
    print("Feedback leakage checks passed",flush=True)
    keras.utils.set_random_seed(SEED)
    original=keras.models.load_model(str(initial_dir/"mobilenetv2.keras"),compile=False)
    base=next(layer for layer in original.layers if isinstance(layer,keras.Model) and "mobilenet" in layer.name)
    base.trainable=False
    inputs=keras.Input((160,160,3),dtype="uint8")
    features=base(keras.layers.Rescaling(1/127.5,offset=-1)(inputs),training=False)
    extractor=keras.Model(inputs,features)
    head=keras.models.clone_model(original.get_layer("waste_classifier"))
    head.set_weights(original.get_layer("waste_classifier").get_weights())

    rng=random.Random(SEED)
    by_category={category:[] for category in CATEGORIES}
    for row in manifest["images"]:
        if row["split"]=="train":
            by_category[row["category"]].append(row)
    selected=[]
    for category,rows in by_category.items():
        rng.shuffle(rows)
        selected.extend(rows if category in {"METAL","CARDBOARD","GLASS","PLASTIC","PAPER","FOOTWEAR","CLOTHES"} else rows[:300])
    train_features=[]; train_labels=[]; human_weights=[]
    print("Extracting training features",len(selected),flush=True)
    for offset in range(0,len(selected),32):
        chunk=selected[offset:offset+32]
        batch=np.stack([image_array(ROOT/row["path"]) for row in chunk])
        train_features.append(extractor(batch,training=False).numpy())
        train_labels.extend(CATEGORIES.index(row["category"]) for row in chunk)
        human_weights.extend([1.0]*len(chunk))
        # Brown color variation on metal examples teaches appearance diversity.
        metal=[row for row in chunk if row["category"]=="METAL"]
        if metal:
            arrays=[]
            for row in metal:
                im=load_rgb(ROOT/row["path"])
                tint=Image.new("RGB",im.size,(135,85,50))
                im=Image.blend(im,tint,0.22)
                arrays.append(array_from_image(im))
            train_features.append(extractor(np.stack(arrays),training=False).numpy())
            train_labels.extend([CATEGORIES.index("METAL")]*len(arrays))
            human_weights.extend([1.0]*len(arrays))
    mosaic_parents={}
    mosaic_rng=random.Random(SEED+1)
    for category,rows in by_category.items():
        mosaics=[]
        parents=[]
        for _ in range(64):
            sources=mosaic_rng.sample(rows,4)
            canvas=Image.new("RGB",(160,160),(127,127,127))
            for index,source in enumerate(sources):
                tile=Image.fromarray(array_from_image(load_rgb(ROOT/source["path"]),80))
                canvas.paste(tile,((index%2)*80,(index//2)*80))
            mosaics.append(np.asarray(canvas))
            parents.append([source["path"] for source in sources])
        mosaic_parents[category]=parents
        for offset in range(0,len(mosaics),32):
            train_features.append(extractor(np.stack(mosaics[offset:offset+32]),training=False).numpy())
        train_labels.extend([CATEGORIES.index(category)]*len(mosaics))
        human_weights.extend([1.0]*len(mosaics))
    for row in feedback:
        samples=np.stack(feedback_images(load_rgb(ROOT/row["path"]),64))
        for offset in range(0,len(samples),32):
            train_features.append(extractor(samples[offset:offset+32],training=False).numpy())
        train_labels.extend([CATEGORIES.index(row["category"])]*len(samples))
        human_weights.extend([2.0]*len(samples))
    train_x=np.concatenate(train_features); train_y=np.array(train_labels)
    counts=np.bincount(train_y,minlength=len(CATEGORIES))
    weights=np.array(human_weights,dtype=np.float32)*np.array([len(train_y)/(len(CATEGORIES)*counts[label]) for label in train_y],dtype=np.float32)

    validation_rows=[row for row in manifest["images"] if row["split"]=="validation"]
    val_features=[]
    for offset in range(0,len(validation_rows),32):
        batch=np.stack([image_array(ROOT/row["path"]) for row in validation_rows[offset:offset+32]])
        val_features.append(extractor(batch,training=False).numpy())
    val_x=np.concatenate(val_features)
    val_y=np.array([CATEGORIES.index(row["category"]) for row in validation_rows])
    options=tf.data.Options(); options.threading.private_threadpool_size=2
    training=tf.data.Dataset.from_tensor_slices((train_x,train_y,weights)).shuffle(len(train_y),seed=SEED).batch(32).with_options(options)
    validation=tf.data.Dataset.from_tensor_slices((val_x,val_y)).batch(32).with_options(options)
    head.compile(optimizer=keras.optimizers.Adam(1e-4),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    history=head.fit(training,validation_data=validation,epochs=18,verbose=2,shuffle=False,callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss",patience=4,restore_best_weights=True)])
    model=keras.Model(inputs,head(features),name="siklus_human_feedback")
    model.save(OUT/"mobilenetv2.keras")
    (OUT/"training-history.json").write_text(json.dumps({
        "history":history.history,"best_epoch":int(np.argmin(history.history["val_loss"]))+1,
        "seed":SEED,"training_files":[row["path"] for row in selected],
        "feedback":feedback,"feedback_variants_each":64,"feedback_sample_weight":2.0,
        "metal_tint_weight":0.22,"class_counts":counts.tolist(),
        "mosaic_parents":mosaic_parents,"mosaics_per_class":64,
        "initial_model_sha256":hashlib.sha256(Path(str(initial_dir/"mobilenetv2.keras")).read_bytes()).hexdigest(),
    },indent=2)+"\n")
    for row in feedback:
        manifest["images"].append({
            "path":row["path"],"category":row["category"],"sha256":hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest(),
            "group":"human-"+hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest(),
            "split":"train","source":"human_feedback",
        })
        manifest["counts"]["train"][row["category"]]+=1
    (OUT/"split.json").write_text(json.dumps(manifest,indent=2)+"\n")
    detector=BottleDetector()
    print("Evaluating fixed validation policy",flush=True)
    initial_metadata=json.loads((initial_dir/"model.json").read_text())
    validation_sha=hashlib.sha256(json.dumps(validation_rows,sort_keys=True).encode()).hexdigest()
    cache_path=initial_dir/"validation-metrics.json"
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if (cache.get("model_sha256")==initial_metadata["sha256"]
            and cache.get("inference")==initial_metadata.get("inference")
            and cache.get("validation_sha256")==validation_sha):
        before=cache["metrics"]
        print("Verified baseline validation cache",flush=True)
    else:
        before=evaluate_validation(original,validation_rows,detector)
    after=evaluate_validation(model,validation_rows,detector)
    gate=after["balanced_accuracy"]>=before["balanced_accuracy"]-0.005
    for category in ("METAL","CARDBOARD","GLASS","FOOTWEAR","PAPER"):
        gate=gate and after["classification_report"][category]["f1-score"]>=before["classification_report"][category]["f1-score"]-0.01
    gate=gate and after["classification_report"]["CLOTHES"]["f1-score"]>=before["classification_report"]["CLOTHES"]["f1-score"]
    (OUT/"validation-comparison.json").write_text(json.dumps({"baseline":before,"candidate":after,"gate":bool(gate)},indent=2)+"\n")
    print("Validation gate",gate,flush=True)
    if not gate:
        return
    old_metadata=json.loads((initial_dir/"model.json").read_text())
    sha=hashlib.sha256((OUT/"mobilenetv2.keras").read_bytes()).hexdigest()
    metadata=dict(old_metadata,sha256=sha,modelVersion="mobilenetv2-feedback-"+sha[:12])
    (OUT/"model.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (OUT/"validation-metrics.json").write_text(json.dumps({
        "model_sha256":sha,"inference":metadata.get("inference"),
        "validation_sha256":validation_sha,"metrics":after},indent=2)+"\n")
    from evaluate import evaluate
    evaluate(ROOT,OUT/"split.json",OUT)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model-dir",type=Path,
        default=Path("artifacts/before-metal-feedback") if Path("artifacts/before-metal-feedback").exists() else Path("artifacts"))
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output
    train(args.initial_model_dir)
