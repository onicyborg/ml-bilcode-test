# Glass/plastic material refinement experiment

This experiment investigates the misclassified glass-bottle bucket image without adding that image to training.

Candidate changes are fixed before validation comparison:
- Increase input resolution from 160 to 224 pixels.
- Warm-start the backbone and classifier from the ten-class model.
- Unfreeze the last 45 backbone layers, keeping BatchNormalization frozen.
- Use learning rate 8e-6, six epochs maximum, early stopping on validation loss.
- Retain all training glass, plastic, and metal samples and rehearse up to 250 samples from each other class.
- Add whole TACO scenes only when all annotations map to the same target material and annotated objects occupy at least 5% of the frame. Keep every scene in the same split/group as its object crops.
- Apply on-the-fly flips, mild rotation, zoom, and contrast augmentation. Use the same aspect-preserving bilinear resizing in training and serving.

Promotion is conditional on a validation gate set before evaluating the candidate. Balanced accuracy may lose at most one percentage point, glass-to-plastic mistakes must decrease, and plastic F1 may lose at most two points. The held-out test result must still exceed the assessment's 80% target. Test labels are not used in training or checkpoint selection. The existing test has been used for prior model reporting, and should be supplemented with new independent field photos before making general deployment claims.

The user-provided bucket photo remains a development example, outside train/validation/test assignments. Its result is checked after model selection. No filename-specific inference rule or lookup table is used.

Run from ml/:

```sh
uv run --locked python refine_materials.py
```

Artifacts and full stage history are saved in experiments/materials224. This model still emits one category. It does not identify every material in a mixed scene, and confidence remains uncalibrated.

The 224px candidate was rejected by the validation gate: glass-to-plastic errors increased from 15 to 20. Its weights are retained for review and were not promoted.

## Conditional bottle proposals

A second candidate retains the existing trained classifier and first locates multiple bottles with OpenCV Zoo's pretrained COCO YOLOX-s detector. Only images whose whole-image prediction is glass, plastic, or metal are eligible. Detections require score >= 0.5, NMS IoU 0.45, and a minimum crop side of 16 pixels, with at most six bottles. At least two detections are required. A material must receive at least two votes and at least two-thirds of the crop votes, and also win the mean crop probability. Otherwise the original whole-image result is retained. No file identity, report description, or manually supplied category affects inference.

This fixed policy is evaluated on validation before test. The gate allows at most 0.5 percentage points of balanced-accuracy loss, no increase in glass-to-plastic errors, and at most one point of plastic F1 loss. Test data is evaluated only if this gate passes. The reported bucket photo is a separate development example.

Run uv run --locked python evaluate_bottles.py to reproduce the policy comparison against a baseline-only classifier. Store provenance and the detector checksum with the model metadata. Confidence for routed images is the mean class score across detected bottles, not a calibrated probability that the label is correct.

The fixed bottle policy passed validation. Whole-image and pipeline accuracies were both 92.69% on validation and 92.35% on the same 2,484 test images. Glass-to-plastic errors did not increase (15 validation, 9 test). The reported bucket photo changed from PLASTIC to GLASS after backend normalization. This example is a development case and is excluded from the held-out metrics. The active release preserves its classifier training history and adds inference-policy provenance; it does not claim that YOLOX was trained on the waste dataset.

Restore the files from artifacts/whole-image-10class and restart FastAPI to roll back. To reproduce the policy release, run evaluate_bottles.py and promote_bottle_policy.py after reviewing the measured gate and development example.
