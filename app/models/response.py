from pydantic import BaseModel


class Prediction(BaseModel):
    class_name: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: list[Prediction]