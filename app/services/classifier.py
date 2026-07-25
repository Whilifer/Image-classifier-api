from PIL import Image

import torch

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from app.models.response import Prediction, PredictionResponse

class Classifier:

    def __init__(self):

        self.model = None
        self.transform = None
        self.categories = None

    def load(self):

        weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(
            weights=weights
        )

        self.model.eval()

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

    def predict(
        self,
        image: Image.Image
    ):

        tensor = self.preprocess(image)

        output = self.inference(tensor)

        return self.postprocess(output)

    def preprocess(
            self,
            image: Image.Image
    ):
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor

    def inference(
            self,
            tensor: torch.Tensor
    ):
        with torch.inference_mode():
            output = self.model(tensor)

        return output

    def postprocess(
        self,
        output: torch.Tensor
    ):

        probabilities = torch.softmax(
            output,
            dim=1
        )

        values, indices = torch.topk(
            probabilities,
            k=3
        )

        predictions = []

        for confidence, index in zip(
            values[0],
            indices[0]
        ):

            predictions.append(
                Prediction(
                    class_name=self.categories[index.item()],
                    confidence=round(
                        confidence.item(),
                        4
                    )
                )
            )

        return PredictionResponse(
            predictions=predictions
    )