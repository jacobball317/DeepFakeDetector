import cv2
import numpy as np
import os

# Check if a frame is blurry by computing the variance of its Laplacian
def is_blurry(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()  # Calculate Laplacian variance
    return variance < threshold  # Return True if below threshold (blurry)

# Check if a frame has appropriate brightness
def is_well_lit(frame, low=40, high=220):
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Average brightness
    return low < brightness < high  # Return True if brightness is in range

# Resize image with preserved aspect ratio and padding to fit target size
def resize_with_aspect_ratio(img, size=(750, 750)):
    h, w = img.shape[:2]
    target_w, target_h = size

    scale = min(target_w / w, target_h / h)  # Calculate scaling factor
    new_w, new_h = int(w * scale), int(h * scale)  # New dimensions after resizing

    resized = cv2.resize(img, (new_w, new_h))  # Resize image

    # Calculate padding to reach desired size
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    # Add black padding to the resized image
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded

# Extract good quality frames from a video
def extract_frames(video_path, num_frames=5, save_frames_path=None, video_name=None):
    cap = cv2.VideoCapture(video_path)  # Open video file
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

    # Generate indices to sample from evenly across the video
    frame_indices = np.linspace(0, total_frames - 1, num_frames * 2, dtype=int)

    frames = []
    accepted = 0  # Counter for accepted frames

    for i in range(total_frames):
        ret, frame = cap.read()  # Read next frame
        if not ret:
            break  # End of video

        if i in frame_indices:
            # Check if the frame is clear and well-lit
            if not is_blurry(frame) and is_well_lit(frame):
                padded = resize_with_aspect_ratio(frame, (750, 750))  # Resize with padding
                gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                normalized = gray / 255.0  # Normalize pixel values
                frames.append(normalized)
                accepted += 1

                # Save the frame if a save path is provided
                if save_frames_path and video_name:
                    frame_filename = f"{video_name}_frame_{i}.jpg"
                    frame_path = os.path.join(save_frames_path, frame_filename)
                    cv2.imwrite(frame_path, padded)

                # Stop if enough frames have been accepted
                if accepted >= num_frames:
                    break

    cap.release()  # Release the video capture object
    return frames

# Calculate features from differences between consecutive frames
def extract_features(frames):
    if len(frames) < 2:
        return np.array([])  # Return empty array if not enough frames

    # Compute mean absolute difference between consecutive frames
    return np.array([
        np.mean(cv2.absdiff((frames[i] * 255).astype(np.uint8),
                            (frames[i - 1] * 255).astype(np.uint8)))
        for i in range(1, len(frames))
    ])

# Process all videos in a folder and extract their features
def process_folder(folder_path, save_frames_base_path, label=""):
    features = {}
    save_frames_path = os.path.join(save_frames_base_path, f"{label}Extracted")
    os.makedirs(save_frames_path, exist_ok=True)  # Create directory if it doesn't exist

    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Process only video files
            video_path = os.path.join(folder_path, filename)
            print(f"Processing {label} video: {filename}")
            frames = extract_frames(
                video_path,
                num_frames=25,  # Number of frames to extract
                save_frames_path=save_frames_path,
                video_name=filename
            )
            if frames:
                video_features = extract_features(frames)  # Extract features from frames
                features[filename] = video_features
    return features

# Main function to run the pipeline
def main():
    base_path = os.path.dirname(os.path.abspath(__file__))  # Get current directory
    fake_videos_path = os.path.join(base_path, "Fake")  # Path to fake videos
    real_videos_path = os.path.join(base_path, "Real")  # Path to real videos
    save_frames_base_path = os.path.join(base_path, "ExtractedFrames")  # Save location
    os.makedirs(save_frames_base_path, exist_ok=True)

    # Process fake videos and extract their features
    print("Processing fake videos...")
    fake_features = process_folder(fake_videos_path, save_frames_base_path, label="Fake")

    # Process real videos and extract their features
    print("Processing real videos...")
    real_features = process_folder(real_videos_path, save_frames_base_path, label="Real")

    print("Processing complete.")

# Entry point of the script
if __name__ == "__main__":
    main()
