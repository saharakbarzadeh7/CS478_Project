import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Define the model for LGG vs HGG classification
def create_glioma_classification_model(input_shape):
    model = Sequential()

    # Convolutional layers
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

input_shape = (160, 210, 100, 1)
model = create_glioma_classification_model(input_shape)

# Print model summary
model.summary()

# Assume weighted_images is the dataset of shape (num_samples, 160, 210, 100, 1)
#  labels is a binary array (0 for LGG, 1 for HGG)
import numpy as np
num_samples = 10
weighted_images = np.load('/Users/sgutta/Desktop/Brain_Tumor_Data/MICCAI_BraTS2020_TrainingData/inputs.npy')
labels = np.load('/Users/sgutta/Desktop/Brain_Tumor_Data/MICCAI_BraTS2020_TrainingData/label.npy')


# Train the model
history = model.fit(weighted_images, labels, batch_size=2, epochs=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(weighted_images, labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
