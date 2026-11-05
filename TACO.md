# TACO experiment (historical three-class model)

The source archive is TACO v0.1, downloaded and extracted by the candidate into ../temp-image/TACO/TACO. Its data/annotations.json contains 715 images and 2,152 object annotations. Original image files and generated crops are excluded from Git.

For current training use MULTICLASS.md. To reconstruct the historical TACO crops, run from ml/:

```sh
uv run --locked python prepare_taco.py
uv run --locked python compare_taco.py
```

Preparation corrects EXIF orientation before interpreting polygon coordinates. Visual review found inconsistent bounding boxes in this old archive. Crop boundaries are recomputed from segmentation polygons and discrepancies are counted in the manifest. It rejects crops smaller than 40 pixels on either side. The mapping is explicitly listed in prepare_taco.py and copied to the manifest. Paper includes clear paper/cardboard categories. Ambiguous coated cartons, foam, multilayer packaging, and unspecified containers are excluded.

All crops from one image are linked into a group. Decoded-pixel hashes and difference hashes connect similar crops and source photos to other samples, including the original Kaggle collection. Existing Kaggle assignments remain fixed. TACO groups touching multiple Kaggle splits are excluded. New groups are assigned with seed 42 at approximately 70/15/15 proportions. Source grouping takes priority over exact per-class ratios.

Training uses the same preprocessing and frozen MobileNetV2 backbone as the baseline, and inverse-frequency weights balance the three output classes. Validation loss selects the checkpoint. Test metrics are reported separately for Kaggle and TACO crops. TACO contains only eight Food waste annotations before size filtering, so organic results from TACO alone have little statistical support.

This experiment trains on cropped annotated objects. It does not establish accuracy for entire mixed-trash scenes. A single-category classifier cannot describe every material in a photo. Do not treat a changed prediction on one user example as evidence of general improvement.

The archive includes toolkit code under its LICENSE. Image permissions must be checked separately using the original image sources before redistribution. Source images are kept outside version control.

Promotion rule is set before comparison. The candidate must improve TACO validation accuracy, match or improve combined validation accuracy, and lose at most two percentage points on the original Kaggle validation set. The held-out test result is reported after model selection and must still meet the assessment's 80% target. If the validation rule fails, the original model stays active.

The completed experiment retained 897 crops (769 plastic, 120 paper, 8 organic). Crop bounds were recomputed for 983 mapped annotations; 87 mapped crops were excluded for size. The candidate met the predeclared validation gates and was promoted to artifacts. The original model is preserved in artifacts/baseline-v1. Training history and comparison metrics are stored in experiments/taco.
