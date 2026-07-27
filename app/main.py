from fastapi import FastAPI
from app.lifespan import lifespan
from app.routers.predict import router as predict_router
from app.exceptions import (
    global_exception_handler
)

app = FastAPI(
    title="IMG-Classificator",
    version="1.0.0",
    lifespan=lifespan
)

app.add_exception_handler(
    Exception,
    global_exception_handler
)

app.include_router(predict_router)