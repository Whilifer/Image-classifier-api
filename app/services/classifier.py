import time

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from app.config import settings
from app.logger import logger
from app.models.response import Prediction, PredictionResponse


class Classifier:
    def __init__(self):

        # self.mlflow = None #не используется

        self.model = None
        self.transform = None
        self.categories = None

    # def set_mlflow(self, mlflow): #не используется
    #     self.mlflow = mlflow

    def load(self):

        weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(weights=weights)

        self.model.eval()

        self.device = settings.DEVICE

        self.model.to(self.device)

        logger.info("Model %s loaded on %s", settings.MODEL_NAME, self.device)

        self.transform = weights.transforms()

        self.categories = weights.meta["categories"]

    # def predict(
    #     self,
    #     image: Image.Image
    # ):

    #     tensor = self.transform(image)

    #     tensor = tensor.unsqueeze(0)

    #     with torch.no_grad():

    #         output = self.model(tensor)

    #         probabilities = torch.softmax(
    #             output,
    #             dim=1
    #         )

    #     index = probabilities.argmax(
    #         dim=1
    #     ).item()

    #     confidence = probabilities[
    #         0,
    #         index
    #     ].item()

    #     return {
    #         "class": self.categories[index],
    #         "confidence": confidence
    #     }

    def predict(self, image: Image.Image):
        start = time.perf_counter()

        tensor = self.preprocess(image)

        output = self.inference(tensor)

        result = self.postprocess(output)

        elapsed = (time.perf_counter() - start) * 1000

        # не используется
        # self.mlflow.log_prediction(
        #     model_name="ResNet18",
        #     device=self.device,
        #     confidence=confidence,
        #     inference_time_ms=elapsed,
        #     predicted_class=predicted_class
        # )

        logger.info("Prediction finished in %.2f ms", elapsed)

        return result

    def preprocess(self, image: Image.Image):
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor

    def inference(self, tensor: torch.Tensor):
        with torch.inference_mode():
            output = self.model(tensor)

        return output

    def postprocess(self, output: torch.Tensor):

        probabilities = torch.softmax(output, dim=1)

        values, indices = torch.topk(probabilities, k=settings.TOP_K)

        predictions = []

        for confidence, index in zip(values[0], indices[0], strict=False):
            predictions.append(
                Prediction(
                    class_name=self.categories[index.item()],
                    confidence=round(confidence.item(), 4),
                )
            )

        return PredictionResponse(predictions=predictions)
