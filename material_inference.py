"""Whole-image classification with conditional multi-bottle material consensus."""

from collections import Counter

import numpy as np

from preprocessing import array_from_image


def prediction(image, model, categories, size, detector=None, baseline=None):
    scores = (model(array_from_image(image, size)[None], training=False).numpy()[0]
              if baseline is None else np.asarray(baseline))
    materials = {categories.index(name) for name in ("GLASS", "PLASTIC", "METAL") if name in categories}
    if detector is None or int(np.argmax(scores)) not in materials:
        return scores, False
    boxes = detector.boxes(image)
    if len(boxes) < 2:
        return scores, False
    crops = np.stack([array_from_image(image.crop(bounds), size) for bounds, _ in boxes])
    crop_scores = model(crops, training=False).numpy()
    votes = Counter(int(index) for index in crop_scores.argmax(axis=1))
    winner, count = votes.most_common(1)[0]
    if winner not in materials or count < 2 or count / len(boxes) < 2 / 3:
        return scores, False
    combined = crop_scores.mean(axis=0)
    if int(np.argmax(combined)) != winner:
        return scores, False
    return combined, True
