import torch


class Settings:

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    TOP_K = 3

    MODEL_NAME = "ResNet18"


settings = Settings()