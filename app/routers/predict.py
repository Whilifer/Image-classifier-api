from fastapi import APIRouter, Depends, UploadFile, File
from app.dependencies import get_classifier
from app.services.classifier import Classifier
from io import BytesIO
from PIL import Image
from app.models.response import PredictionResponse
from app.config import settings
from app.logger import logger

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.get("/health")
def health(
    classifier: Classifier = Depends(get_classifier)
):

    return {
        "status": "ok",
        "model_loaded": classifier.model is not None,
        "device": settings.DEVICE,
        "model": settings.MODEL_NAME
    }

@router.post("/", response_model=PredictionResponse)
async def classify(
    file: UploadFile = File(...),
    classifier: Classifier = Depends(get_classifier)
):
    image = Image.open(
        BytesIO(await file.read())
    ).convert("RGB")

    logger.info(
        "Image received: %dx%d",
        image.width,
        image.height
    )

    result = classifier.predict(image)
    return result