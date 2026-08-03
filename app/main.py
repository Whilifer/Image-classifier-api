from fastapi import FastAPI
from app.lifespan import lifespan
from app.routers.predict import router as predict_router
from app.exceptions import global_exception_handler
# from app.services.mlflow_service import MLflowService

app = FastAPI(title="IMG-Classificator", version="1.0.0", lifespan=lifespan)

app.add_exception_handler(Exception, global_exception_handler)

# mlflow_service = MLflowService()
# classifier = Classifier(
#     mlflow_service=mlflow_service
# )

app.include_router(predict_router)
