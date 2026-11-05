"""Internal image classification API used by the NestJS backend."""

from contextlib import asynccontextmanager
from io import BytesIO
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock
import warnings

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from tensorflow import keras

from preprocessing import CATEGORIES, load_rgb
from bottle_detector import BottleDetector, MODEL_PATH
from material_inference import prediction

ARTIFACTS = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "artifacts"))
MAX_BYTES = 5 * 1024 * 1024
inference_lock = Lock()
logger = logging.getLogger("siklus.ml")


@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    app.state.version = None
    app.state.detector = None
    try:
        metadata = json.loads((ARTIFACTS / "model.json").read_text())
        path = ARTIFACTS / "mobilenetv2.keras"
        if (not metadata["classes"] or len(set(metadata["classes"])) != len(metadata["classes"])
                or not set(metadata["classes"]).issubset(CATEGORIES) or metadata["imageSize"] not in (160, 224)):
            raise ValueError("Incompatible model metadata")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Model checksum mismatch")
        model = keras.models.load_model(path, compile=False)
        if model.input_shape[1:3] != (metadata["imageSize"], metadata["imageSize"]):
            raise ValueError("Model input does not match metadata")
        model(np.zeros((1, metadata["imageSize"], metadata["imageSize"], 3), dtype=np.uint8), training=False)
        if model.output_shape[-1] != len(metadata["classes"]):
            raise ValueError("Model output does not match category metadata")
        inference = metadata.get("inference")
        if inference:
            if inference.get("policy") != "bottle_consensus_v1":
                raise ValueError("Unsupported inference policy")
            if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != inference["detectorSha256"]:
                raise ValueError("Detector checksum mismatch")
            app.state.detector = BottleDetector()
        app.state.policy = inference["policy"] if inference else "whole_image"
        app.state.image_size = metadata["imageSize"]
        app.state.classes = metadata["classes"]
        app.state.model = model
        app.state.version = metadata["modelVersion"]
    except (OSError, ValueError, KeyError, cv2.error):
        logger.error("Model could not be loaded. Check MODEL_DIR and exported artifacts.")
    yield
    app.state.model = None


app = FastAPI(
    title="Siklus Waste Classification",
    description="Internal API. Classifies a single dominant waste type across ten supported categories.",
    version="1.0.0",
    lifespan=lifespan,
)


class Prediction(BaseModel):
    category: str = Field(enum=list(CATEGORIES))
    confidence: float = Field(ge=0, le=1)
    modelVersion: str


class Error(BaseModel):
    detail: str


@app.get("/health", summary="Check model readiness", responses={503: {"model": Error}})
def health():
    if app.state.model is None:
        raise HTTPException(503, "Model is not ready")
    return {"status": "ready", "modelVersion": app.state.version, "categories": app.state.classes, "imageSize": app.state.image_size, "inferencePolicy": app.state.policy}


def classify(content):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = load_rgb(BytesIO(content))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(422, "Upload a valid JPEG or PNG with at most 25 million pixels")
    with inference_lock:
        probabilities, _ = prediction(image, app.state.model, app.state.classes,
                                     app.state.image_size, app.state.detector)
    index = int(np.argmax(probabilities))
    return Prediction(category=app.state.classes[index], confidence=float(probabilities[index]),
                      modelVersion=app.state.version)


@app.post("/predict", response_model=Prediction, summary="Classify an uploaded waste photo",
          responses={413: {"model": Error}, 422: {"model": Error}, 503: {"model": Error}})
async def predict(photo: UploadFile = File(description="JPEG or PNG, maximum 5 MiB")):
    try:
        content = await photo.read(MAX_BYTES + 1)
    finally:
        await photo.close()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "Photo exceeds 5 MiB")
    if app.state.model is None:
        raise HTTPException(503, "Model is not ready")
    return await run_in_threadpool(classify, content)
