from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.classifier import Classifier

classifier = Classifier()


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading resnet18 model...")

    classifier.load()

    print("Model resnet18 loaded")

    yield

    print("Shutdown")