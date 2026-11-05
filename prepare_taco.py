"""Create conservative TACO object crops and a grouped combined manifest."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageOps

from dataset import fingerprint, SEED

ROOT = Path("../temp-image")
SOURCE = ROOT / "TACO/TACO/data"
OUTPUT = ROOT / "taco-crops"
ARTIFACTS = Path("experiments/taco")
MAPPING = {
    "Food waste": "ORGANIC",
    "Other plastic bottle": "PLASTIC",
    "Plastic drink bottle": "PLASTIC",
    "Plastic bottle cap": "PLASTIC",
    "Disposable plastic cup": "PLASTIC",
    "Other plastic cup": "PLASTIC",
    "Plastic lid": "PLASTIC",
    "Other plastic": "PLASTIC",
    "Plastic Film": "PLASTIC",
    "Six pack rings": "PLASTIC",
    "Garbage bag": "PLASTIC",
    "Other plastic wrapper": "PLASTIC",
    "Single-use carrier bag": "PLASTIC",
    "Polypropylene bag": "PLASTIC",
    "Other plastic container": "PLASTIC",
    "Plastic glooves": "PLASTIC",
    "Plastic utensils": "PLASTIC",
    "Plastic straw": "PLASTIC",
    "Magazine paper": "PAPER",
    "Tissues": "PAPER",
    "Wrapping paper": "PAPER",
    "Normal paper": "PAPER",
    "Paper bag": "PAPER",
    "Paper straw": "PAPER",
    "Toilet tube": "PAPER",
    "Egg carton": "PAPER",
    "Corrugated carton": "PAPER",
}

def signature(image):
    digest = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
    return digest, *fingerprint(image)

def prepare(baseline_path=Path("artifacts/baseline-v1/split.json")):
    annotation_path = SOURCE / "annotations.json"
    source = json.loads(annotation_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    by_image = defaultdict(list)
    categories = {row["id"]: row["name"] for row in source["categories"]}
    for annotation in source["annotations"]:
        if categories[annotation["category_id"]] in MAPPING:
            by_image[annotation["image_id"]].append(annotation)

    records = [dict(row, source="kaggle") for row in baseline["images"]]
    signatures = []
    for row in records:
        with Image.open(ROOT / row["path"]) as image:
            signatures.append(signature(ImageOps.exif_transpose(image).convert("RGB")))
    links = []
    skipped = Counter()
    for source_image in source["images"]:
        annotations = by_image[source_image["id"]]
        if not annotations:
            continue
        with Image.open(SOURCE / source_image["file_name"]) as image:
            photo = ImageOps.exif_transpose(image).convert("RGB")
            if photo.size != (source_image["width"], source_image["height"]):
                raise ValueError("Image orientation does not match annotations")
            photo_index = len(records)
            records.append({"source": "taco-photo", "path": "TACO/TACO/data/" + source_image["file_name"]})
            signatures.append(signature(photo))
            for annotation in annotations:
                polygons = annotation.get("segmentation", [])
                if not isinstance(polygons, list) or not polygons:
                    skipped["missing_polygon"] += 1
                    continue
                xs = [x for polygon in polygons for x in polygon[0::2]]
                ys = [y for polygon in polygons for y in polygon[1::2]]
                if not xs or not ys:
                    skipped["invalid_polygon"] += 1
                    continue
                left, top = max(0, math.floor(min(xs))), max(0, math.floor(min(ys)))
                right, bottom = min(photo.width, math.ceil(max(xs))), min(photo.height, math.ceil(max(ys)))
                x, y, width, height = annotation["bbox"]
                if max(abs(left-x), abs(top-y), abs(right-x-width), abs(bottom-y-height)) > 2:
                    skipped["bbox_recomputed_from_polygon"] += 1
                if min(right - left, bottom - top) < 40:
                    skipped["small_crop"] += 1
                    continue
                crop = photo.crop((left, top, right, bottom))
                category = MAPPING[categories[annotation["category_id"]]]
                output = OUTPUT / f"{source_image['id']}-{annotation['id']}.jpg"
                output.parent.mkdir(parents=True, exist_ok=True)
                crop.save(output, quality=95)
                links.append((photo_index, len(records)))
                records.append({
                    "source": "taco", "path": output.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "category": category, "source_image": source_image["file_name"],
                    "annotation_id": annotation["id"], "original_category": categories[annotation["category_id"]],
                })
                signatures.append(signature(crop))

    parents = list(range(len(records)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    def union(a, b):
        parents[find(a)] = find(b)
    base_groups = {}
    for index, row in enumerate(baseline["images"]):
        group = row["group"]
        if group in base_groups:
            union(index, base_groups[group])
        else:
            base_groups[group] = index
    for a, b in links:
        union(a, b)
    for i, (digest, horizontal, vertical) in enumerate(signatures):
        for j in range(i):
            other, h, v = signatures[j]
            if digest == other or ((horizontal ^ h).bit_count() <= 4 and (vertical ^ v).bit_count() <= 4):
                union(i, j)
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[find(index)].append(row)
    rng = random.Random(SEED)
    combined = []
    for index, group in groups.items():
        splits = {row["split"] for row in group if row["source"] == "kaggle"}
        if len(splits) > 1:
            skipped["taco_crops_with_conflicting_base_splits"] += sum(row["source"] == "taco" for row in group)
            combined.extend(dict(row, group=f"baseline-{row['group']}") for row in group if row["source"] == "kaggle")
            continue
        split = next(iter(splits)) if splits else rng.choices(
            ["train", "validation", "test"], weights=[0.70, 0.15, 0.15], k=1)[0]
        combined.extend(dict(row, split=split, group=f"combined-{index}")
                        for row in group if row["source"] != "taco-photo")

    counts = {split: dict(Counter(row["category"] for row in combined if row["split"] == split))
              for split in ("train", "validation", "test")}
    taco_counts = {split: dict(Counter(row["category"] for row in combined if row["split"] == split and row["source"] == "taco"))
                   for split in counts}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED, "counts": counts, "taco_counts": taco_counts, "mapping": MAPPING,
        "source_annotations_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        "skipped": dict(skipped), "images": combined,
    }
    (ARTIFACTS / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"total": len(combined), "counts": counts, "taco_counts": taco_counts, "skipped": dict(skipped)}, indent=2))

if __name__ == "__main__":
    prepare()
