"""
Central settings for PhytoGuard AI.

Every path and training setting lives here, so the notebook, the API and
the retraining code all use the same values.
"""

import os

# Project root (the folder that contains src/, models/ and data/).
# Can be overridden with an environment variable, which is useful in Docker.
BASE_DIR = os.environ.get(
    "PHYTOGUARD_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Model files
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phytoguard_ai_final.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
RETRAIN_LOG_PATH = os.path.join(MODEL_DIR, "retrain_log.json")
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")

# Data folders (one subfolder per class inside each)
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")      # replay data for retraining
TEST_DIR = os.path.join(DATA_DIR, "test")        # held-out images for model comparison
RETRAIN_DIR = os.path.join(DATA_DIR, "retrain")  # sample bulk upload used in the notebook
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")   # new images uploaded by users

# Image settings (must match the notebook)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Retraining settings
RETRAIN_THRESHOLD = 200      # automatic trigger after this many new images
RETRAIN_EPOCHS = 3
RETRAIN_LEARNING_RATE = 1e-5
FINE_TUNE_LAYERS = 30        # top MobileNetV2 layers unfrozen during fine-tuning
TOP_K = 3                    # number of predictions returned to the user
