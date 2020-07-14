import numpy as np
from nn_model.nn_model import data_management as dm
from nn_model.nn_model import model
import cv2
import logging

_logger = logging.getLogger(__name__)


def make_single_prediction(im_path: str):

    img = dm.read_image(im_path)
    img = cv2.resize(img, dsize=(224, 224))
    img = img.astype(float)
    img *= 1./255
    img = np.expand_dims(img, axis=0)

    modell = model.cnn_model()
    proba = modell.predict(img)

    result = (proba > 0.5).astype("int32")

    if result == 1:
        label = "Emergency"
    else:
        label = "Not Emergency"
    _logger.info(
        f"Input: {im_path}"
        f"Predicted Probability: {proba}"
        f"Result of Model: {label}"
    )

    return result, proba
