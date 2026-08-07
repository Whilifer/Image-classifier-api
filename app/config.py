import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_NAME: str = "ResNet18"

    TOP_K: int = 3

    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024

    MAX_BATCH_SIZE: int = 16

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # class Config:
    #     env_file = ".env"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
