# ML implementation

- Use Python 3.12 and uv.lock. Keep source, manifest, model hash, and evaluation results traceable.
- Split image groups before augmentation. Train and select checkpoints with train/validation only. Report the held-out test accuracy without tuning against test results.
- Keep category order and decoding identical between training and serving. Avoid custom Keras layers that require unsafe model deserialization.
- Keep uploaded photos and their bytes out of logs. Bound upload size, decoded image dimensions, and inference concurrency.
- Validate the saved model, API error behavior, and one NestJS integration path when changing inference. Record limitations and actual measured results.
