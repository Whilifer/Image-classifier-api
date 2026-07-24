from fastapi import FastAPI
from app.lifespan import lifespan
from app.routers.predict import router as predict_router

app = FastAPI(
    title="IMG-Classificator",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(predict_router)