# Train-only material corrections

This experiment incorporates two labels explicitly confirmed by the user: the mixed rusty-can scene is METAL as its dominant requested category, and the bottle bucket is GLASS. These two files were listed in feedback.json for this stage. The current file also contains the subsequent paper correction described in PAPER_REFINEMENT.md. They are training corrections and regression checks, not independent evidence of field accuracy.

The source files are excluded from Git with the rest of temp-image. Their hashes and train-only group assignments are retained in the candidate manifest. Preparation checks exact hashes and difference hashes against validation and test before training to prevent these corrections from entering held-out evaluation.

The existing MobileNetV2 backbone is frozen to preserve the learned representation. The classifier head is warm-started and retrained with learning rate 1e-4 for up to 18 epochs, with validation-loss early stopping. Rehearsal retains all training metal, cardboard, glass, and plastic samples and up to 300 examples per other category. A mild brown tint augments metal examples to broaden color variation. Each human correction contributes 64 deterministic geometric/brightness/contrast variants with sample weight 2. Inverse-frequency weights balance the classes.

The deployed bottle-consensus policy remains part of validation. Promotion requires balanced accuracy to lose at most 0.5 percentage points, and metal, cardboard, and glass F1 to lose at most one point each. Test accuracy must still meet the assessment's 80% target. All thresholds are fixed before reviewing the candidate results. Regression checks on the two human corrections must also pass, but are reported separately from accuracy.

Historical invocation from ml/ (the current trainer adds the paper stage; this command no longer reproduces the earlier two-photo experiment):

```sh
uv run --locked python train_feedback.py --initial-model-dir artifacts/before-metal-feedback
```

Artifacts are saved under experiments/feedback-metal. The manifest contains physical source-file counts; training-history.json also records augmented sample counts, feedback variants, sample weighting, and source filenames. Confidence remains an uncalibrated score. This update is not a general mixed-material detector, and the user-labeled dominant category does not imply that every object in the scene is metal.

The accepted candidate reached 92.75% on the same 2,484 test images, versus 92.35% before the update. Both feedback regressions pass after backend normalization. The two corrections are now training examples and must not be counted as independent field-test successes.

The default starting checkpoint is the archived pre-feedback release when available. Pass --initial-model-dir explicitly to reproduce the initial conditions or select a different starting release.
