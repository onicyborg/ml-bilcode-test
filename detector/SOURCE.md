# Bottle detector provenance

The model is OpenCV Zoo YOLOX-s pretrained for the 80 COCO object classes. Only bottle proposals are used. It does not predict glass or plastic; Siklus's trained classifier determines material from the proposed image crops.

Model URL: https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx

Reference implementation: https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox

License is retained in LICENSE (Apache 2.0). bottle_detector.py adapts the documented letterbox preprocessing and anchor decoding. The model's SHA-256 is recorded in the active metadata and experiment comparison.

Re-download with curl into detector/yolox.onnx and verify its SHA-256 against model.json before serving. The user's complaint photo was not added to Siklus training or held-out evaluation.
