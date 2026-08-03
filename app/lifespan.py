from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.classifier import Classifier
from app.logger import logger
from app.config import settings
# from app.services.mlflow_service import MLflowService

classifier = Classifier()
# mlflow = MLflowService() #не используется


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(f"Loading {settings.MODEL_NAME} model...")

    classifier.load()

    # classifier.set_mlflow(mlflow) #не используется

    logger.info(f"Model {settings.MODEL_NAME} loaded")

    yield

    logger.info("Shutdown")
