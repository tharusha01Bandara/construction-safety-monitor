from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/best.pt"
    OUTPUT_DIR: str = "outputs"
    
    # Model predict param
    MODEL_PREDICT_CONF: float = 0.15

    # Confidence Thresholds
    PERSON_CONF_THRESHOLD: float = 0.40
    HELMET_CONF_THRESHOLD: float = 0.20
    VEST_CONF_THRESHOLD: float = 0.10

    # Person dimension filtering
    PERSON_MIN_AREA: int = 12000
    PERSON_MIN_WIDTH: int = 40
    PERSON_MIN_HEIGHT: int = 80

    # Matching logic constants
    HELMET_UPPER_RATIO: float = 0.35
    VEST_TORSO_TOP_RATIO: float = 0.20
    VEST_TORSO_BOTTOM_RATIO: float = 0.80
    VEST_OVERLAP_THRESHOLD: float = 0.15

    # Review threshold
    MANUAL_REVIEW_CONFIDENCE: float = 0.45

settings = Settings()
