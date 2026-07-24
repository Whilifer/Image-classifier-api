from fastapi import APIRouter, Depends, UploadFile, File
from app.dependencies import get_classifier
from app.services.classifier import Classifier
from io import BytesIO
from PIL import Image

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
        "model_name": classifier.model.__class__.__name__
    }

@router.post("/")
async def classify(
    file: UploadFile = File(...),
    classifier: Classifier = Depends(get_classifier)
):
    image = Image.open(
        BytesIO(await file.read())
    ).convert("RGB")

    result = classifier.predict(image)
    return result