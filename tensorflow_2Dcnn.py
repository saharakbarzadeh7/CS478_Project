import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Define a simple 2D CNN model
def build_2d_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='softmax')  # Output probabilities for HGG/LGG
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

X = np.load('D:/CSUSM-Computer Science/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/inputs_2d.npy')
y = np.load('D:/CSUSM-Computer Science/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/label_2d.npy')
X_train = X[:25800, :, :, :]
y_train = y[:25800]
print(X_train.shape)
print(y_train.shape)

X_val = np.load('D:/CSUSM-Computer Science/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/inputs.npy')
y_val = np.load('D:/CSUSM-Computer Science/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/label.npy')
X_test = X_val[-111:, :, :, :]
y_test = y_val[-111:]
print(X_test.shape)
print(y_test.shape)

# Build and train the model
model = build_2d_cnn((160, 210, 1))
model.fit(X_train, y_train, epochs=1, validation_split=0.2, batch_size=32)

# 3. Predict grade for a 3D MRI scan using 2D slices
def predict_scan_3d(mri_scan, model):
    depth = mri_scan.shape[-1]
    predictions = []

    for i in range(depth):
        # Extract the i-th 2D slice
        slice_2d = mri_scan[:, :, i]
        slice_2d = slice_2d.reshape(1, 160, 210, 1)  # Add batch and channel dimensions

        # Predict the grade for this slice
        prob = model.predict(slice_2d, verbose=0)[0]  # Get probabilities
        pred = np.argmax(prob)  # Get predicted class (0 or 1)
        predictions.append(pred)

    return predictions

# 4. Perform majority voting
def majority_voting(predictions):
    return max(set(predictions), key=predictions.count)
"""
# Gaussian Weighting for Slice Importance
num_slices = 100
sigma = num_slices / 4
slice_indices = np.arange(num_slices) #ensures the sum of all weights equals 1-creates an array from 0 t0 99
gaussian_weights = np.exp(-((slice_indices - num_slices // 2) ** 2) / (2 * sigma ** 2))#the formula for findinf the center slice(50)
gaussian_weights /= gaussian_weights.sum()  # Normalize weights

# Weighted Voting
def weighted_voting(predictions, weights):
    weighted_sum = np.sum(predictions * weights[:, np.newaxis], axis=0)  # Weighted sum of probabilities
    return np.argmax(weighted_sum)  # Return class with highest probability
"""
# Predict grades for slices and apply majority voting
final_grade = np.zeros(111)
for i in range(111):
    slice_predictions = predict_scan_3d(X_test[i, :, :, :], model)
    final_grade[i] = majority_voting(slice_predictions)

accuracy = accuracy_score(y_test, final_grade)
print(f"Accuracy: {accuracy * 100:.2f}%")
