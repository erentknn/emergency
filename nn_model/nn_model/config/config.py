import os
import logging
import sys

PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, ".."))
DATASET_DIR = os.path.join(PACKAGE_ROOT, "dataset")

IMAGES_DIR = os.path.join(DATASET_DIR, "images")

TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, "trained_models")
TEMP_IMAGE_DIR = os.path.join(TRAINED_MODEL_DIR, "temp_images")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

IMAGE_SIZE = 224
BATCH_SIZE = 32

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s -" "%(funcName)s:%(lineno)d - %(message)s"
)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler
