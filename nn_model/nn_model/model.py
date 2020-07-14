from nn_model.nn_model.config import config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten
import os
import logging

_logger = logging.getLogger(__name__)


def cnn_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="valid", activation="relu", input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    model.trainable = False
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.load_weights(os.path.join(config.TRAINED_MODEL_DIR, "weights_loss.hdf5"))

    return model


if __name__ == "__main__":

    main_model = cnn_model()
    main_model.summary()
