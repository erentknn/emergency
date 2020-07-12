from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1].resolve()
DATASET_DIR = PACKAGE_ROOT / "dataset"

IMAGES_DIR = DATASET_DIR / "images"

TRAIN_DATA_DIR = "train.csv"
TEST_DATA_DIR = "test.csv"

TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
TEMP_IMAGE_DIR = TRAINED_MODEL_DIR / "temp_images"

IMAGE_SIZE = 224
BATCH_SIZE = 32
