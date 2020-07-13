import pathlib
import logging
import sys
import nn_model

PACKAGE_ROOT = pathlib.Path(nn_model.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / "dataset"

IMAGES_DIR = DATASET_DIR / "images"

TRAIN_DATA_DIR = "train.csv"
TEST_DATA_DIR = "test.csv"

TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
TEMP_IMAGE_DIR = TRAINED_MODEL_DIR / "temp_images"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

IMAGE_SIZE = 224
BATCH_SIZE = 32

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s -" "%(funcName)s:%(lineno)d - %(message)s"
)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler