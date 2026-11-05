# Siklus waste classification

Python 3.12, TensorFlow/Keras MobileNetV2, and FastAPI. Run commands from ml/. Dependencies are pinned in uv.lock.

```sh
uv sync --locked
uv run --locked uvicorn api:app --host 127.0.0.1 --port 8000
```

NestJS uses ML_URL=http://localhost:8000. Keep FastAPI on the internal network. Documentation is at http://localhost:8000/docs. GET /health reports model readiness, version, and supported output categories.

POST /predict accepts multipart photo, containing JPEG or PNG, up to 5 MiB and 25 million decoded pixels. It returns category, confidence (0–1), and modelVersion. Invalid images return 422, oversized photos 413, and an unavailable model 503. API startup verifies the model hash and checks the output dimension against model.json. Override the artifact directory with MODEL_DIR.

## Categories

| API identifier | Meaning |
| --- | --- |
| ORGANIC | Biological waste |
| PLASTIC | Plastic |
| PAPER | Paper |
| GLASS | Glass, combining three source colors |
| METAL | Metal |
| CARDBOARD | Cardboard |
| CLOTHES | Clothes |
| FOOTWEAR | Shoes |
| BATTERY | Batteries |
| OTHER | Source dataset trash class |

OTHER does not guarantee rejection of every unknown object. A photo containing multiple materials still receives a single predicted category.

## Reproduce training

Download the [Kaggle Garbage Classification dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) and retain all twelve source folders under ../temp-image. Keep the candidate-downloaded [TACO archive](https://zenodo.org/records/3354286) under ../temp-image/TACO/TACO. Sources and generated crops are excluded from Git. Review source license terms before redistributing image files.

```sh
uv run --locked python prepare_multiclass.py
uv run --locked python train.py --manifest experiments/multiclass/split.json --output experiments/multiclass --fine-tune-epochs 4
```

See MULTICLASS.md for category mapping, split isolation, source corrections, and fine-tuning details. The training code checks source hashes and split grouping before fitting. Validation loss selects between the trained classifier head and the fine-tuned model. Evaluation on the held-out test occurs after selection. Retraining may yield small numerical differences across TensorFlow builds and CPUs.

## Evaluate and promote

```sh
uv run --locked python evaluate.py
```

Current results are in artifacts/evaluation.json, classification-report.txt, source-evaluation.json, test-predictions.json, and confusion-matrix.png. Training history records the selected stage and epoch. Copy the selected model, model.json, split.json, and all evaluation artifacts together when promoting, then restart FastAPI. The prior three-class model is preserved for rollback.

The test set measures labeled waste examples and TACO object crops. It does not establish accuracy on entire mixed-trash scenes. Confidence is an uncalibrated softmax score. User review is necessary. Uploaded photos do not automatically retrain the model.


## Bottle material handling

The optional bottle_consensus_v1 policy first identifies multiple bottles with the pretrained OpenCV Zoo YOLOX detector, then applies the trained material classifier to those crops. Only confident multi-bottle agreement can replace a whole-image glass/plastic/metal prediction. Model metadata records the policy and detector checksum. Keep detector/yolox.onnx and its license with the deployment. See MATERIAL_REFINEMENT.md and detector/SOURCE.md for validation, provenance, and reproduction.

Restart FastAPI after a model/policy promotion. GET /health reports modelVersion and inferencePolicy so the running version can be verified. Test a new report to see new predictions, since existing records are historical snapshots.

## Human corrections

The current classifier also learns from four explicit training corrections in feedback.json (metal, glass, paper, and clothes). Put those user-supplied files under ../temp-image to reproduce this stage. Run `uv run --locked python train_feedback.py --initial-model-dir artifacts/before-clothes-feedback --output experiments/feedback-clothes`. Held-out metrics exclude all four files. See CLOTHES_REFINEMENT.md for rehearsal, leakage checks, predeclared validation gates, actual results and limitations. PAPER_REFINEMENT.md and FEEDBACK_TRAINING.md document preceding stages.


## Repository artifact scope

This repository includes the active classifier, detector, held-out evaluation and current validation/regression records in artifacts/. Historical snapshots and intermediate experiments remain local and are excluded to avoid publishing redundant checkpoints. Commands referring to archived starting checkpoints require those optional local archives; training a new candidate can explicitly use --initial-model-dir artifacts with a fresh --output directory. Raw datasets remain external; preserve their documented provenance when obtaining them.
