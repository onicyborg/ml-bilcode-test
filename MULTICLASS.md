# Ten-category model

The expanded model predicts one dominant category per photo. The API identifiers are ORGANIC, PLASTIC, PAPER, GLASS, METAL, CARDBOARD, CLOTHES, FOOTWEAR, BATTERY, and OTHER.

Glass combines brown-glass, green-glass, and white-glass from the Kaggle collection. Clothes and footwear are kept separate because shoe materials vary. OTHER uses the source trash class. It is a trained category, and does not guarantee recognition of all unknown objects. Mixed waste may still be assigned a single unsuitable label.

## Reproduction

Run in ml/ with Python 3.12 and uv:

```sh
uv sync --locked
uv run --locked python prepare_multiclass.py
uv run --locked python train.py --manifest experiments/multiclass/split.json --output experiments/multiclass --fine-tune-epochs 4
```

Sources are the user-downloaded Kaggle Garbage Classification dataset and TACO archive. Keep the Kaggle folders and TACO/TACO/data under ../temp-image. Review each source's terms before redistributing images. Source images and derived crops are excluded from Git.

Preparation maps all twelve Kaggle folders to ten labels. It excludes ambiguous groups of visually similar source images with conflicting labels. For TACO, it uses an explicit conservative mapping, corrects EXIF orientation, and computes crop bounds from polygon annotations because this archive's bounding boxes are inconsistent. Crops smaller than 40 pixels on either side are excluded. Similar source photos and all their crops stay in one split. Crops that would connect different base splits are excluded.

The manifest retains 15,501 Kaggle images and 1,240 TACO crops, for 16,741 samples. Train has 11,753 samples, validation 2,504, and test 2,484. These are new ten-class assignments. Their headline metrics are not directly comparable to the earlier three-class test.

## Training and selection

Preprocessing is shared with inference: EXIF orientation, RGB conversion, aspect-preserving padding to 160×160, and MobileNetV2 scaling. Frozen ImageNet features are extracted once for training images and horizontal flips. A dropout-regularized classifier head is trained with inverse-frequency class weights. Validation loss selects its checkpoint through early stopping.

The second stage unfreezes the last 30 backbone layers while keeping BatchNormalization layers frozen. It uses learning rate 1e-5, on-the-fly flips, mild rotations, and contrast changes, for up to four epochs. The fine-tuned checkpoint is retained only if its validation loss improves over the classifier-head checkpoint. The held-out test is evaluated after that selection.

Training history, full test predictions, per-source reports, confusion matrix, hashes, and manifest are written under experiments/multiclass. Promotion requires test accuracy of at least 80% and matching category metadata. Model files and metadata must be copied together. Earlier artifacts are preserved for rollback.

## Application integration

The database migration 005_expand_waste_categories adds seven enum values while preserving old reports. API validation, Swagger, frontend filters, and labels use the same ten identifiers. FastAPI reads the output order from model.json and checks its output dimension. Restart FastAPI after promoting a new model. Existing report predictions are historical snapshots and are not recomputed automatically.

The selected model reached 92.35% accuracy on 2,484 held-out samples, with macro F1 0.9073. Fine-tuning epoch three was selected by validation loss. The external glass-bottle bucket photo remains a PLASTIC prediction and is recorded separately as a field failure.
