from nn_model.predict import make_single_prediction
from nn_model.config import config


def test_make_prediction_sample():
    filename = "0.jpg"
    expected_classification = 1

    result = make_single_prediction(im_path=config.IMAGES_DIR/filename)

    assert result is not None
    assert result == expected_classification

    filename2 = "176.jpg"

    expected_classification2 = 1

    result2 = make_single_prediction(im_path=config.IMAGES_DIR/filename2)

    assert result2 is not None
    assert result2 == expected_classification2


test_make_prediction_sample()
