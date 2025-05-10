import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import math

#Image augmentation module (flip, rotation, translation)
def get_augmenter(min_area):
    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential([
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
    ])

#Visualization function: show original + weak + strong augmentations
def visualize_augmentations(num_images, train_slices):
    # Pick N images and add grayscale channel
    #images = train_slices[:num_images].astype(np.float32)
    indices = np.random.choice(len(train_slices), num_images, replace=False)
    images = train_slices[indices].astype(np.float32)

    images = np.expand_dims(images, axis=-1)  # (num_images, 160, 210, 1)

    # Apply augmentations
    augmented_images = zip(
        images,
        get_augmenter(**classification_augmentation)(images),
        get_augmenter(**contrastive_augmentation)(images),
        get_augmenter(**contrastive_augmentation)(images),
    )

    row_titles = [
        "Original:",
        "Weakly augmented:",
        "Strongly augmented:",
        "Strongly augmented:",
    ]

    # Plot grid
    plt.figure(figsize=(num_images * 2.2, 4 * 2.2), dpi=100)
    for column, image_row in enumerate(augmented_images):
        for row, image in enumerate(image_row):
            plt.subplot(4, num_images, row * num_images + column + 1)
            plt.imshow(tf.squeeze(image), cmap="gray")
            if column == 0:
                plt.title(row_titles[row], loc="left")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

#Example usage
if __name__ == "__main__":
    # Load 2D slices
    train_slices = np.load('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/train_slices.npy')

    # Augmentation configs (no brightness or zoom)
    classification_augmentation = {
        "min_area": 0.8,  # Weak
    }
    contrastive_augmentation = {
        "min_area": 0.5,  # Strong
    }

    # Visualize
    visualize_augmentations(num_images=8, train_slices=train_slices)
