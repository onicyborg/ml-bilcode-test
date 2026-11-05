"""Build a reproducible, stratified manifest with similar images kept together."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from PIL import Image, ImageOps

FOLDERS = {"biological": "ORGANIC", "plastic": "PLASTIC", "paper": "PAPER"}
SEED = 42


def fingerprint(image):
    gray = image.convert("L")
    horizontal = gray.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    vertical = gray.resize((8, 9), Image.Resampling.LANCZOS).tobytes()
    return (
        sum((horizontal[y * 9 + x] > horizontal[y * 9 + x + 1]) << (y * 8 + x)
            for y in range(8) for x in range(8)),
        sum((vertical[y * 8 + x] > vertical[(y + 1) * 8 + x]) << (y * 8 + x)
            for y in range(8) for x in range(8)),
    )


def prepare(root, output):
    records, hashes, invalid = [], [], []
    for folder, category in FOLDERS.items():
        for path in sorted((root / folder).iterdir()):
            if not path.is_file():
                continue
            try:
                with Image.open(path) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
                    digest.update(rgb.tobytes())
                    hashes.append((digest.hexdigest(), *fingerprint(rgb)))
            except (OSError, ValueError):
                invalid.append(path.relative_to(root).as_posix())
                continue
            records.append({"path": path.relative_to(root).as_posix(), "category": category,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    if len(records) < 150 or set(FOLDERS.values()) != {row["category"] for row in records}:
        raise ValueError("Dataset must contain at least 150 usable images across all requested classes")

    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    matched = 0
    for i, (digest, horizontal, vertical) in enumerate(hashes):
        for j in range(i):
            other, h, v = hashes[j]
            if digest == other or ((horizontal ^ h).bit_count() <= 4 and (vertical ^ v).bit_count() <= 4):
                parents[find(i)] = find(j)
                matched += 1

    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row)
    by_class = defaultdict(list)
    excluded = []
    for index, group in groups.items():
        categories = {row["category"] for row in group}
        if len(categories) != 1:
            excluded.extend(row["path"] for row in group)
            continue
        by_class[categories.pop()].append((index, group))

    rng = random.Random(SEED)
    manifest = []
    for category in dict.fromkeys(FOLDERS.values()):
        class_groups = by_class[category]
        rng.shuffle(class_groups)
        size = sum(len(group) for _, group in class_groups)
        target = round(size * 0.15)
        counts = Counter()
        for group_id, group in class_groups:
            split = "test" if counts["test"] < target else "validation" if counts["validation"] < target else "train"
            counts[split] += len(group)
            manifest.extend(dict(row, split=split, group=group_id) for row in group)
        if not all(counts[split] for split in ("train", "validation", "test")):
            raise ValueError(f"Insufficient independent groups for {category}")

    counts = {split: dict(Counter(row["category"] for row in manifest if row["split"] == split))
              for split in ("train", "validation", "test")}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "seed": SEED, "matched_pairs": matched, "invalid": invalid,
        "conflicting_labels": excluded, "counts": counts, "images": manifest,
    }, indent=2) + "\n")
    print(json.dumps({"matched_pairs": matched, "invalid": len(invalid),
                      "excluded": len(excluded), "counts": counts}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../temp-image"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/split.json"))
    args = parser.parse_args()
    prepare(args.data, args.output)
