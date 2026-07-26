from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.classifier import Classifier
from app.logger import logger
from app.config import settings

classifier = Classifier()


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(f"Loading {settings.MODEL_NAME} model...")

    classifier.load()

    logger.info(f"Model {settings.MODEL_NAME} loaded")

    yield

    logger.info("Shutdown")