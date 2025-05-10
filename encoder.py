#first encoder
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the encoder architecture
def get_encoder(width=128, input_shape=(160, 210, 1)):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),  # For 2D grayscale MRI
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
        ],
        name="encoder",
    )

'''
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def get_encoder(input_shape=(160, 210, 1), weights_path= "D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/RadImageNet_models/RadImageNet-ResNet50_notop.h5"):
    inputs = layers.Input(shape=input_shape)

    # Convert grayscale to 3-channel
    x = layers.Concatenate()([inputs, inputs, inputs])  # → shape (160, 210, 3)

    # Load model architecture (no weights yet)
    base_model = ResNet50(include_top=False, weights=None, input_shape=(160, 210, 3), pooling='avg')

    # Load RadImageNet weights
    base_model.load_weights(weights_path)

    base_model.trainable = False  # optional: freeze for now

    x = base_model(x)
    x = layers.Dense(128, activation="relu")(x)  # project to SimCLR feature space

    model = models.Model(inputs=inputs, outputs=x, name="radimagenet_encoder")
    return model


#replace encoder with pretrained ResNet50
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def get_encoder(input_shape=(160, 210, 1)):
    inputs = layers.Input(shape=input_shape)

    # Repeat grayscale channel to simulate 3-channel input
    x = layers.Concatenate()([inputs, inputs, inputs])  # Shape becomes (160, 210, 3)

    # Load pretrained ResNet50 without top classification layer
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(160, 210, 3), pooling="avg")
    base_model.trainable = False  # Freeze weights for stability

    x = base_model(x)
    x = layers.Dense(128, activation="relu")(x)  # Optional projection to 128-dim feature

    model = models.Model(inputs=inputs, outputs=x, name="resnet50_encoder")
    return model
'''