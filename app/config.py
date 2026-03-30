from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/best.pt"
    OUTPUT_DIR: str = "outputs"
    CONFIDENCE_THRESHOLD: float = 0.5
    MIN_PERSON_AREA: int = 1000  # Filter out tiny detections
    REVIEW_CONFIDENCE_THRESHOLD: float = 0.6  # Flag for review if below this

settings = Settings()
