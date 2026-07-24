from PIL import Image

import torch

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


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

    def predict(
        self,
        image: Image.Image
    ):

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        with torch.no_grad():

            output = self.model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

        index = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            index
        ].item()

        return {
            "class": self.categories[index],
            "confidence": confidence
        }