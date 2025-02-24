import os
import cv2
import numpy as np
import tensorflow as tf

# Constants
IMAGE_SIZE = (224, 224)  # Model input size
MODEL_PATH = "Model/Example.pth"  # Path to the trained model
THRESHOLD = 0.5  # Classification threshold

# Load the trained Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess an image
def preprocess_frame(frame):
    """
    Convert frame to grayscale, resize, normalize, and expand dimensions for model input.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, IMAGE_SIZE)  # Resize to match model input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values (0 to 1)
    final_frame = np.expand_dims(normalized_frame, axis=(0, -1))  # Add batch & channel dimensions
    return final_frame

# Function to process a video and predict CHD
def analyze_video(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Unable to open video file: {video_path}")
        return

    frame_predictions = []
    print("🔍 Processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop when there are no more frames

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Predict
        prediction = model.predict(processed_frame, verbose=0)[0][0]  # Get prediction score
        frame_predictions.append(prediction)

    cap.release()

    # Calculate average prediction score
    avg_prediction = np.mean(frame_predictions) if frame_predictions else 0
    print(f"📊 Average Prediction Score: {avg_prediction:.4f}")

    # Final classification
    if avg_prediction > THRESHOLD:
        print("🛑 Prediction: CHD Detected (VSD)")
    else:
        print("✅ Prediction: No CHD Detected")

# Get video file path from user
video_path = input("Enter the path to the video file: ").strip()
analyze_video(video_path)