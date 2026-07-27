import torch


class Settings:

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    TOP_K = 3

    MODEL_NAME = "ResNet18"

    MAX_UPLOAD_SIZE = 10 * 1024 * 1024


settings = Settings()