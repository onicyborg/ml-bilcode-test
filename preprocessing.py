from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 160
MAX_PIXELS = 25_000_000
CATEGORIES = ("ORGANIC", "PLASTIC", "PAPER", "GLASS", "METAL", "CARDBOARD", "CLOTHES", "FOOTWEAR", "BATTERY", "OTHER")


def load_rgb(source):
    with Image.open(source) as image:
        if image.format not in {"JPEG", "PNG"} or image.width * image.height > MAX_PIXELS:
            raise ValueError("Expected a JPEG or PNG with at most 25 million pixels")
        return ImageOps.exif_transpose(image).convert("RGB")


def array_from_image(image, image_size=IMAGE_SIZE):
    resized = ImageOps.pad(image, (image_size, image_size), Image.Resampling.BILINEAR, color=(127, 127, 127))
    return np.asarray(resized, dtype=np.uint8)


def image_array(source, image_size=IMAGE_SIZE):
    return array_from_image(load_rgb(source), image_size)


def decode_photo(content: bytes, image_size=IMAGE_SIZE):
    return image_array(BytesIO(content), image_size)
