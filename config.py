"""
Configuration for the VinBigData Chest X-ray project.

What this file does:
- Defines common paths used across the project
- Stores simple, readable configuration dictionaries for dataset, models, training, and logging
- Ensures required directories exist
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "name": "VinBigData_Chest_Xray",
    # Default subset size for quick experiments (training on a smaller subset)
    "subset_size_gb": 2,
    "image_formats": ["PNG", "JPEG"],
    "target_resolution": (512, 512),
    "num_classes": 14,  # Number of abnormality classes
    "class_names": [
        "Aortic enlargement",
        "Atelectasis", 
        "Calcification",
        "Cardiomegaly",
        "Consolidation",
        "ILD",
        "Infiltration",
        "Lung Opacity",
        "Nodule/Mass",
        "Other lesion",
        "Pleural effusion",
        "Pleural thickening",
        "Pneumothorax",
        "Pulmonary fibrosis"
    ]
}

# Model configuration
MODEL_CONFIG = {
    "classification": {
        # Simpler, beginner-friendly backbone
        "backbone": "resnet18",
        "num_classes": 14,
        "input_size": (512, 512),
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 100
    },
    "detection": {
        "model_type": "yolov8",
        "input_size": (640, 640),
        "batch_size": 8,
        "learning_rate": 1e-3,
        "epochs": 100,
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45
    }
}

# Training configuration
TRAINING_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "num_workers": 4,
    "pin_memory": True,
    "mixed_precision": True,
    "gradient_clip": 1.0,
    "early_stopping_patience": 10,
    "save_best_only": True
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "train": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.1,
        "rotation": 15,
        "brightness_contrast": 0.2,
        "elastic_transform": 0.1,
        "grid_distortion": 0.1
    },
    "val": {
        "horizontal_flip": False,
        "vertical_flip": False,
        "rotation": 0,
        "brightness_contrast": 0,
        "elastic_transform": 0,
        "grid_distortion": 0
    }
}

# Evaluation metrics
EVALUATION_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1", "auc"],
    "detection": ["map", "map50", "map75", "precision", "recall"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"]
}
