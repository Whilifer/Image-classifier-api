from io import BytesIO
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.dependencies import get_classifier
from app.logger import logger
from app.models.response import BatchPredictionResponse, PredictionResponse
from app.services.classifier import Classifier

router = APIRouter(prefix="/predict", tags=["Prediction"])


async def load_image(
    file: UploadFile,
) -> Image.Image:
    content = await file.read()

    if not file.content_type.startswith("image/"):
        logger.warning("Invalid file type: %s", file.content_type)
        raise HTTPException(
            status_code=400,
            detail="File must be an image",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large",
        )

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        logger.exception("Cannot read image")
        raise HTTPException(
            status_code=400,
            detail="Invalid image",
        ) from None

    logger.info(
        "Image received: %dx%d",
        image.width,
        image.height,
    )

    return image


@router.get("/health")
def health(classifier: Classifier = Depends(get_classifier)):
    return {
        "status": "ok",
        "model_loaded": classifier.model is not None,
        "device": settings.DEVICE,
        "model": settings.MODEL_NAME,
    }


@router.post("/", response_model=PredictionResponse)
async def classify(
    file: UploadFile = File(...), classifier: Classifier = Depends(get_classifier)
):
    image = await load_image(file)

    result = classifier.predict(image)
    return result


@router.post("/batch", response_model=BatchPredictionResponse)
async def classify_batch(
    classifier: Annotated[Classifier, Depends(get_classifier)],
    files: Annotated[List[UploadFile], File(...)],
):
    if len(files) > settings.MAX_BATCH_SIZE:
        logger.warning(
            "Batch size exceeded: %d files (max %d)",
            len(files),
            settings.MAX_BATCH_SIZE,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.MAX_BATCH_SIZE} files are allowed",
        )

    images = []

    for file in files:
        image = await load_image(file)
        images.append(image)
    result = classifier.predict_many(images)
    return result
