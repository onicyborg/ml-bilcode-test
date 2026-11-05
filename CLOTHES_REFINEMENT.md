# Clothing-pile correction

The user-confirmed CLOTHES photo is train-only feedback alongside the earlier METAL, GLASS and PAPER corrections. None is counted as independent field testing. Exact and perceptual duplicate checks run against held-out data before training.

The active paper-correction checkpoint is the starting model. Its frozen MobileNetV2 backbone is retained; the classifier head receives all training clothes, paper, footwear, glass, metal, plastic and cardboard examples, plus up to 300 per remaining class. Existing same-class mosaics and correction augmentations are retained.

Predeclared validation gates: clothing F1 must not decrease; balanced accuracy may lose at most 0.5 percentage points; paper, metal, glass, cardboard and footwear F1 may lose at most one percentage point. Test is evaluated only after validation passes. All four normalized correction regressions must pass before promotion.

Initial command: `uv run --locked python train_feedback.py --initial-model-dir artifacts --output experiments/feedback-clothes`.

After promotion reproduce using `artifacts/before-clothes-feedback` as the starting directory. The previous checkpoint is retained there. Existing benchmark reuse limits conclusions about new field photos; one-category classification remains approximate for cluttered scenes.

## Application regression

All four train-only corrections pass the frontend upload → NestJS JPEG normalization → ML → R2 → PostgreSQL flow, with category filtering checked. The new clothing-pile image changes from PAPER (uncalibrated score 0.7571) to CLOTHES (score 0.9974). Invalid photo input remains HTTP 422. Temporary application services and generated test data were cleaned up.

## Validation result

All predeclared gates passed. CLOTHES F1 is 0.98812 (previous 0.98809). PAPER F1 decreases from 0.87647 to 0.87087, within the predeclared one-point tolerance; this tradeoff is retained explicitly rather than claiming every category improved. GLASS F1 is 0.90554 (previous 0.89803), and METAL is 0.82456 (previous 0.82558). Epoch 3 was selected by validation loss.

## Promoted release

Held-out accuracy: 93.32% on 2,484 unchanged test paths, compared with 93.00% previously. Active model: mobilenetv2-feedback-b5d5aef88ddd. The prior checkpoint is archived under artifacts/before-clothes-feedback. Restart the ML process to load the new release, then create a new report; existing predictions are historical values.
