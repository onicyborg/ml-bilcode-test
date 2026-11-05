"""Bottle proposals from OpenCV Zoo's pretrained COCO YOLOX model."""

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).parent / "detector/yolox.onnx"


class BottleDetector:
    def __init__(self, path=MODEL_PATH):
        cv2.setNumThreads(2)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        grids, strides = [], []
        for stride in (8, 16, 32):
            x, y = np.meshgrid(np.arange(640 // stride), np.arange(640 // stride))
            grid = np.stack((x, y), axis=-1).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((len(grid), 1), stride))
        self.grids = np.concatenate(grids)
        self.strides = np.concatenate(strides)

    def boxes(self, image):
        scale = min(640 / image.height, 640 / image.width)
        pixels = np.asarray(image.convert("RGB"))
        resized = cv2.resize(pixels, (int(image.width * scale), int(image.height * scale)))
        padded = np.full((640, 640, 3), 114, dtype=np.float32)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        self.net.setInput(padded.transpose(2, 0, 1)[None])
        output = self.net.forward()[0]
        centers = (output[:, :2] + self.grids) * self.strides
        sizes = np.exp(output[:, 2:4]) * self.strides
        scores = output[:, 4] * output[:, 5 + 39]
        selected = (output[:, 5:].argmax(axis=1) == 39) & (scores >= 0.5)
        boxes = np.concatenate((centers-sizes/2, sizes), axis=1)[selected] / scale
        scores = scores[selected]
        if not len(boxes):
            return []
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, 0.45)
        results = []
        for index in np.asarray(keep).reshape(-1)[:6]:
            x, y, width, height = boxes[index]
            bounds = (max(0, int(x)), max(0, int(y)),
                      min(image.width, int(x+width)), min(image.height, int(y+height)))
            if min(bounds[2]-bounds[0], bounds[3]-bounds[1]) >= 16:
                results.append((bounds, float(scores[index])))
        return results
