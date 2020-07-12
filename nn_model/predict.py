import numpy as np
from tensorflow.keras.preprocessing import image
from nn_model import data_management as dm
from nn_model import model
from pathlib import Path
import cv2
import tensorflow
import time

def make_single_prediction(model, im_path:str):

    img = dm.read_image(im_path)
    img = cv2.resize(img, dsize=(224,224))
    print(img.shape)
    org = img.copy()
    img = img.astype(float)
    img *= 1./255
    img = np.expand_dims(img, axis=0)

    proba = model.predict(img)
    print(proba)
    result = (proba > 0.5).astype("int32")

    if result == 1:
        label = "Emergency"
    else:
        label = "Not Emergency"


if __name__ == "__main__":
    start_time = time.time()
    model = model.cnn_model()
    make_single_prediction(model,
                           im_path="https://devirsaati.com/wp-content/uploads/2020/05/Nissan-EV-Ambulance-Exterior-source.jpg")
    print("time elapsed(model-load): {:.2f}s".format(time.time() - start_time))