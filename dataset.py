import numpy as np
from sklearn.model_selection import train_test_split

# Load the saved numpy arrays
inputs = np.load('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/inputs.npy', mmap_mode='r')
labels = np.load('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/label.npy', mmap_mode='r')
start = np.load('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/start.npy', mmap_mode='r')
end = np.load('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/end.npy', mmap_mode='r')

# Convert to float32 to save memory
inputs = inputs.astype(np.float16)

# Compute mean across last axis (4 sequences → 1 mean value), ensuring dtype is float32
inputs_mean = np.mean(inputs, axis=-1, dtype=np.float16)

# Save the mean dataset
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/inputs_mean.npy', inputs_mean)

print("Shape after computing mean:", inputs_mean.shape)  # Expected: (357, 160, 210, 100)

# Perform stratified train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test, start_train, start_test, end_train, end_test = train_test_split(
    inputs_mean, labels, start, end, test_size=0.3, stratify=labels, random_state=42
)
# Print the size of train and test sets
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# Function to extract 2D slices based on start and end indices
def extract_slices(data, start_indices, end_indices, labels):
    extracted_slices = []
    extracted_labels = []
    
    for i in range(data.shape[0]):  # Iterate over patients
        start_idx = int(start_indices[i])
        end_idx = int(end_indices[i])
        
        # Extract slices between start and end index
        for slice_idx in range(start_idx, end_idx):
            extracted_slices.append(data[i, :, :, slice_idx])  # Shape: (160, 210)
            extracted_labels.append(labels[i])  # Assign same label to all extracted slices
            
    return np.array(extracted_slices, dtype=np.float16), np.array(extracted_labels, dtype=np.int8)

# Extract 2D slices for training and testing
train_slices, train_labels_2d = extract_slices(X_train, start_train, end_train, y_train)
test_slices, test_labels_2d = extract_slices(X_test, start_test, end_test, y_test)

# Save extracted 2D slices
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/train_slices.npy', train_slices)
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/test_slices.npy', test_slices)


# Save updated 2D labels for validation and linear probe model
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/train_labels_2d.npy', train_labels_2d)
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/test_labels_2d.npy', test_labels_2d)

# Print final shapes
print("Train slices shape:", train_slices.shape)  # Expected: (N_train_slices, 160, 210)
print("Train labels shape:", train_labels_2d.shape)  # Should match train_slices count
print("Test slices shape:", test_slices.shape)
print("Test labels shape:", test_labels_2d.shape)