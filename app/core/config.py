"""
config.py — Central configuration for the platform.

Why have a config file?
  → Instead of writing "ecg_model_v1.onnx" in 5 different places,
    we write it ONCE here and import it everywhere.
  → If you rename your model file, you change it HERE only.

We use Pydantic's BaseSettings which can also read from a .env file
or environment variables — great for deploying to the cloud later.
"""

from pydantic_settings import BaseSettings
from pathlib import Path

# Path(__file__) = the location of THIS file (config.py)
# .parent = the folder containing it (core/)
# .parent.parent = one level up (app/)
# .parent.parent.parent = one more level up (ai-inference-platform/)
BASE_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    # ── App Identity ──────────────────────────────────────────────────────────
    APP_NAME: str = "AI Inference Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False  # Set to True during development for extra logs

    # ── Model Paths ───────────────────────────────────────────────────────────
    # Directory where all .onnx model files live
    MODELS_STORE_DIR: str = str(BASE_DIR / "models_store")

    # This is the path to your real ECG ONNX model file.
    # We use BASE_DIR so it works no matter what folder you run the app from.
    ECG_MODEL_V1_PATH: str = str(BASE_DIR / "models_store" / "ecg_model_v1.onnx")

    # v2 path — we'll add a second model later (canary deployment).
    # For now it's the same model, just to show the concept.
    ECG_MODEL_V2_PATH: str = str(BASE_DIR / "models_store" / "ecg_model_v1.onnx")

    # ── Inference Settings ────────────────────────────────────────────────────
    # How much traffic goes to v2? 0.0 = all traffic to v1, 0.1 = 10% to v2.
    # This is how "canary deployments" work: slowly shift traffic to the new model.
    CANARY_TRAFFIC_RATIO: float = 0.0

    # Expected input shape for your ECG model (batch_size, channels, signal_length)
    # Discovered from the model: input shape is (batch_size, 12, 5000)
    ECG_INPUT_LENGTH: int = 5000  # number of signal samples (5000 from the real model)
    ECG_NUM_LEADS: int = 12       # 12-lead ECG

    # ── API Settings ──────────────────────────────────────────────────────────
    API_PREFIX: str = "/api"

    model_config = {
        # If a .env file exists, load settings from it.
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Create a single shared instance — everyone imports THIS object.
# "Singleton pattern": there's only ever ONE settings object in memory.
settings = Settings()
