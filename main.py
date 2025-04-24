# Import system and visualization libraries
import os
import cv2
import numpy as np
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import threading
import pickle

# My custom imports
from face_features import FaceFeatureExtractor  # For detecting faces + getting feature vectors
from temporal_classifier import classify_features  # The neural net that classifies real vs. fake

# Try to set up GPU monitoring (for plotting utilization later)
try:
    from pynvml import *
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)
except:
    gpu_handle = None  # fallback in case GPU isn't available or pynvml isn't installed


# This class tracks CPU, RAM, and GPU usage while the program is running
class SystemMonitor:
    def __init__(self, interval=1):
        self.interval = interval  # how often to record data (in seconds)
        self.cpu = []
        self.ram = []
        self.gpu = []
        self.gpu_mem = []
        self.timestamps = []

        self.running = True
        self.start_time = time.time()

        # Use a background thread to constantly update stats
        self.thread = threading.Thread(target=self.update_data)
        self.thread.start()

    def update_data(self):
        # Every interval, record system stats
        while self.running:
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            self.cpu.append(psutil.cpu_percent())
            self.ram.append(psutil.virtual_memory().percent)

            if gpu_handle:
                util = nvmlDeviceGetUtilizationRates(gpu_handle)
                mem = nvmlDeviceGetMemoryInfo(gpu_handle)
                self.gpu.append(util.gpu)
                self.gpu_mem.append(mem.used / mem.total * 100)
            else:
                self.gpu.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.thread.join()  # wait for background thread to stop

    def plot(self):
        # Draw live charts for CPU, RAM, and GPU usage
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].set_title("CPU & RAM Usage")
        ax[1].set_title("GPU Usage (if available)")

        def animate(i):
            ax[0].clear()
            ax[1].clear()

            ax[0].plot(self.timestamps, self.cpu, label="CPU %")
            ax[0].plot(self.timestamps, self.ram, label="RAM %")
            ax[0].legend()
            ax[0].set_ylim(0, 100)

            ax[1].plot(self.timestamps, self.gpu, label="GPU Util %")
            ax[1].plot(self.timestamps, self.gpu_mem, label="GPU Mem %")
            ax[1].legend()
            ax[1].set_ylim(0, 100)

        ani = animation.FuncAnimation(fig, animate, interval=1000)
        plt.tight_layout()
        plt.show()


# Loads all the .jpg images in a given folder
def load_images_from_folder(folder_path):
    images = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)
            if image is not None:
                images[filename] = image
    return images

# Given a dictionary of images, runs face detection + feature extraction
def process_images(image_dict, extractor):
    features = {}
    for name, img in image_dict.items():
        feature_vector = extractor.process_frame(img)
        if feature_vector is not None:
            features[name] = feature_vector
        else:
            print(f"⚠️ No face detected in {name}")  # just in case the detector misses one
    return features

# Main function that runs the full pipeline
def main():
    monitor = SystemMonitor()  # Start system performance tracking

    try:
        # Set up paths to real/fake image folders
        base_path = os.path.dirname(os.path.abspath(__file__))
        fake_path = os.path.join(base_path, "ExtractedFrames", "FakeExtracted")
        real_path = os.path.join(base_path, "ExtractedFrames", "RealExtracted")

        print("Loading fake images...")
        fake_images = load_images_from_folder(fake_path)

        print("Loading real images...")
        real_images = load_images_from_folder(real_path)

        # Create face extractor (uses MTCNN + Xception internally)
        print("Initializing face feature extractor...")
        extractor = FaceFeatureExtractor()

        print("Extracting features from fake images...")
        fake_features = process_images(fake_images, extractor)

        print("Extracting features from real images...")
        real_features = process_images(real_images, extractor)

        print(f"\n✅ Done! Extracted {len(fake_features)} fake features and {len(real_features)} real features.")

        # Save features to disk so we don’t need to recompute during demo
        with open("real_features.pkl", "wb") as f:
            pickle.dump(real_features, f)
        with open("fake_features.pkl", "wb") as f:
            pickle.dump(fake_features, f)
        print("💾 Saved pre-extracted features to disk.")

        # Train and evaluate the model on those features
        print("\n🧠 Training classifier on extracted features...")
        classify_features(real_features, fake_features)

    finally:
        monitor.stop()  # Stop system monitoring
        print("\n📊 Plotting system utilization during processing...")
        monitor.plot()  # Show performance usage charts


# Make sure the script only runs when called directly
if __name__ == "__main__":
    main()
