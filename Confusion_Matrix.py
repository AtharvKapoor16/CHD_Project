import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Constants
CSV_FILE = "labeled_frames_VSD.csv"
IMAGE_SIZE = (224, 224)  # Match the trained model's input size
BATCH_SIZE = 32
MODEL_PATH = "Models/chd_vs_vsd_model.h5"

# Load CSV file
df = pd.read_csv(CSV_FILE)
image_paths = df['file_path'].tolist()  # Image file paths
labels = df['label'].tolist()  # Corresponding labels (0 = No CHD, 1 = CHD)

# Load the trained model
model = load_model(MODEL_PATH)

# Function to preprocess an image
def preprocess_image(image_path):
    """ Load and preprocess an image for model inference. """
    if not os.path.exists(image_path):
        return np.zeros((224, 224, 1))  # Default blank image if not found
    
    img = load_img(image_path, color_mode="grayscale", target_size=IMAGE_SIZE)  # Load image in grayscale
    img_array = img_to_array(img) / 255.0  # Normalize pixel values (0 to 1)
    return img_array  # Shape: (224, 224, 1)

# Prepare dataset
X = np.array([preprocess_image(path) for path in image_paths])
X = np.expand_dims(X, axis=-1)  # Ensure shape (num_samples, 224, 224, 1)
y_true = np.array(labels)  # Convert labels to NumPy array

# Predict class probabilities
y_pred_probs = model.predict(X, batch_size=BATCH_SIZE, verbose=1)  # Probability scores
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary labels

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print results
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['No CHD', 'CHD']))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CHD', 'CHD'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
