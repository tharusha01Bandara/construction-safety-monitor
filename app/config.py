from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/best.pt"
    OUTPUT_DIR: str = "outputs"
    
    # Model predict param
    MODEL_PREDICT_CONF: float = 0.15

    # Confidence Thresholds
    PERSON_CONF_THRESHOLD: float = 0.40
    HELMET_CONF_THRESHOLD: float = 0.25
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

    # --- TEMPORAL ANALYSIS CONFIG ---
    VIDEO_FRAME_STRIDE: int = 5
    WORKER_MATCH_IOU_THRESHOLD: float = 0.3
    PERSISTENT_UNSAFE_WORKER_FRAMES: int = 3
    PERSISTENT_UNSAFE_SCENE_FRAMES: int = 3
    PERSISTENT_LOW_CONF_FRAMES: int = 3
    OUTPUT_VIDEO_FPS: int = 5

    # --- MODEL EVALUATION CONFIG
    DATASET_YAML_PATH: str = "data.yaml"
    EVAL_SPLIT: str = "val"
    EVAL_IMAGE_SIZE: int = 640
    EVAL_BATCH_SIZE: int = 8

settings = Settings()
