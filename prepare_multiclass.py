"""Prepare all Kaggle classes plus polygon-derived TACO objects."""

from pathlib import Path

import dataset
import prepare_taco

FOLDERS = {
    "biological": "ORGANIC", "plastic": "PLASTIC", "paper": "PAPER",
    "brown-glass": "GLASS", "green-glass": "GLASS", "white-glass": "GLASS",
    "metal": "METAL", "cardboard": "CARDBOARD", "clothes": "CLOTHES",
    "shoes": "FOOTWEAR", "battery": "BATTERY", "trash": "OTHER",
}

MAPPING = dict(prepare_taco.MAPPING)
for name in ("Toilet tube", "Egg carton", "Corrugated carton"):
    MAPPING[name] = "CARDBOARD"
MAPPING.update({
    "Glass bottle": "GLASS", "Glass jar": "GLASS", "Glass cup": "GLASS", "Broken glass": "GLASS",
    "Aluminium foil": "METAL", "Aluminium blister pack": "METAL", "Metal bottle cap": "METAL",
    "Food Can": "METAL", "Aerosol": "METAL", "Drink can": "METAL", "Metal lid": "METAL",
    "Pop tab": "METAL", "Scrap metal": "METAL", "Shoe": "FOOTWEAR", "Battery": "BATTERY",
})


def prepare():
    directory = Path("experiments/multiclass")
    dataset.FOLDERS = FOLDERS
    dataset.prepare(Path("../temp-image"), directory / "base-split.json")
    prepare_taco.ARTIFACTS = directory
    prepare_taco.OUTPUT = Path("../temp-image/multiclass-crops")
    prepare_taco.MAPPING = MAPPING
    prepare_taco.prepare(baseline_path=directory / "base-split.json")


if __name__ == "__main__":
    prepare()
