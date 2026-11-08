from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    app_name: str = Field(default="passport-extractor-api")
    debug: bool = Field(default=False)

    yolo_model_path: str = Field(default="models/passport_obb.onnx")
    ocr_model_path: str = Field(default="models/ocr_extractor.onnx")

    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")
    log_max_bytes: int = Field(default=1_000_000)
    log_backup_count: int = Field(default=5)

    @property
    def yolo_model_abs(self) -> Path:
        return Path(self.yolo_model_path)

    @property
    def ocr_model_abs(self) -> Path:
        return Path(self.ocr_model_path)

    @property
    def log_dir_abs(self) -> Path:
        p = Path(self.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()