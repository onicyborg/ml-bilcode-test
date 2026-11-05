# Paper-pile correction

The user-provided newspaper-pile photo is added to feedback.json with the confirmed PAPER label. It becomes training data and is excluded from held-out accuracy. Earlier METAL and GLASS corrections remain in the training set and regression suite.

This candidate warm-starts the active metal-correction model, retains its frozen backbone, and retrains the classifier head. Rehearsal now includes all source training paper and footwear examples as well as metal, glass, cardboard, and plastic. The other classes retain up to 300 source training examples. Each class receives 64 synthetic 2×2 mosaics built from four same-class training photos to introduce varied multiple-object layouts without changing labels. Mosaic source paths are recorded. Feedback still uses 64 augmented views per image and sample weight 2.

The initial checkpoint and validation sample hash are checked before reusing saved baseline validation metrics. The new candidate is measured with the deployed bottle-consensus policy. Promotion requires paper F1 not to decrease, balanced accuracy to lose at most 0.5 percentage points, and metal/cardboard/glass/footwear F1 to lose at most one point each. Test accuracy must meet the 80% assessment target. All three human-correction regressions must pass after backend normalization. These gates are set before seeing candidate results.

Run in ml/:

```sh
uv run --locked python train_feedback.py --initial-model-dir artifacts/before-paper-feedback --output experiments/feedback-paper
uv run --locked python promote_feedback.py --candidate experiments/feedback-paper --backup before-paper-feedback
```

The archived starting checkpoint is created during promotion. During the initial experiment, --initial-model-dir artifacts selected the then-active metal-correction release. Source images and variants remain excluded from version control. This model still emits one category per report and cannot guarantee accuracy on new mixed or cluttered scenes. Held-out benchmarks have been reused across iterations, so new independent field evaluation remains necessary.

## Recorded result

Validation gates passed. Held-out accuracy is 93.00% on 2,484 unchanged test paths (previous release: 92.75%). Paper validation F1 improved from 0.8735 to 0.8765; validation recall improved from 0.8683 to 0.8922. The three train-only corrections pass JPEG normalization and the full frontend → backend → ML → R2 → PostgreSQL flow, including category filters. The paper correction changed from FOOTWEAR (score 0.6908) to PAPER (score 0.9883). These scores are uncalibrated and do not represent field accuracy. Regression details are in experiments/feedback-paper/correction-regression.json.

Restart the existing ML process after promotion; a running process keeps its previously loaded checkpoint. Existing report predictions remain historical values; submit a new report to run inference with the promoted release.
