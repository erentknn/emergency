from nn_model.nn_model.predict import make_single_prediction
from nn_model.nn_model.config import config
import os


def test_make_prediction_sample():
    filename = "0.jpg"
    expected_classification = 1

    result = make_single_prediction(im_path=os.path.join(config.IMAGES_DIR, f"{filename}"))
    os.path.join(config.IMAGES_DIR, "filename")
    assert result is not None
    assert result[0] == expected_classification

    filename2 = "4.jpg"

    expected_classification2 = 1

    result2 = make_single_prediction(im_path=os.path.join(config.IMAGES_DIR, f"{filename2}"))

    assert result2 is not None
    assert result2[0] == expected_classification2
