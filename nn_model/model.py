import pandas as pd
from nn_model.config import config
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

base_model = ResNet50V2(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="avg")

asd = config.IMAGES_DIR


def cnn_model():

    model = Sequential()

    model.add(base_model)

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    model.trainable = False
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.load_weights("D:/Programlama/Python/Projects/emergency/nn_model/trained_models/model_weights.hdf5")

    return model


if __name__ == "__main__":

    main_model = cnn_model()
    main_model.summary()

