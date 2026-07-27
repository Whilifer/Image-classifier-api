import torch
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    MODEL_NAME: str = "ResNet18"

    TOP_K: int = 3

    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024

    DEVICE: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    class Config:
        env_file = ".env"


settings = Settings()