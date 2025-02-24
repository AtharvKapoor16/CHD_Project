import cv2
import os

def extract_frames_from_video(video_path, output_folder, frame_rate=30, resize_dim=(224, 224)):
    """
    Extract frames from a video, convert to grayscale, resize, and save as images.
    """
    if not os.path.isfile(video_path):
        print(f"Error: {video_path} is not a valid file.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
    print(f"Processing video: {video_path}, FPS: {fps}, Total frames: {total_frames}")

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    frame_count = 0
    extracted_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        if frame_count % max(1, int(fps / frame_rate)) == 0:  # Extract based on frame rate
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            resized_frame = cv2.resize(gray_frame, resize_dim)  # Resize to 224x224

            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
            extracted_frames += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_frames} frames from {video_path}.")

def extract_frames_from_multiple_videos(video_folder, output_folder, frame_rate=30, resize_dim=(224, 224)):
    """
    Process multiple videos in a folder, extracting frames and saving them in separate subfolders.
    """
    if not os.path.isdir(video_folder):
        print(f"Error: {video_folder} is not a valid directory.")
        return

    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print(f"No video files found in {video_folder}.")
        return

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        output_subfolder = os.path.join(output_folder, os.path.splitext(video_file)[0])
        os.makedirs(output_subfolder, exist_ok=True)

        extract_frames_from_video(video_path, output_subfolder, frame_rate, resize_dim)

    print(f"Processed {len(video_files)} videos. Frames saved in {output_folder}.")

# Set paths and run extraction
video_folder = r"Video_Folder"  # Change this to your actual video folder
output_folder = r"Output_Folder"  # Change this to your desired output folder

extract_frames_from_multiple_videos(video_folder, output_folder)