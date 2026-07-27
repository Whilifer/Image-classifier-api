from fastapi import APIRouter, Depends, UploadFile, File
from app.dependencies import get_classifier
from app.services.classifier import Classifier
from io import BytesIO
from PIL import Image
from app.models.response import PredictionResponse
from app.config import settings
from app.logger import logger
from fastapi import HTTPException

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
    content = await file.read()

    if not file.content_type.startswith("image/"):
        logger.warning(
            "Invalid file type: %s",
            file.content_type
        )
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        image = Image.open(
            #BytesIO(await file.read())
            BytesIO(content)
        ).convert("RGB")
    except Exception:
        logger.exception(
            "Cannot read image"
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    #content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )

    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large"
        )

    logger.info(
        "Image received: %dx%d",
        image.width,
        image.height
    )

    result = classifier.predict(image)
    return result