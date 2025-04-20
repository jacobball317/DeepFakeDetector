import torch
from facenet_pytorch import MTCNN
from torchvision import transforms, models
import torch.nn as nn
import cv2
import numpy as np
import timm


class FaceFeatureExtractor:
    def __init__(self, device=None):
        """
        Initializes the FaceFeatureExtractor with a face detector (MTCNN),
        a pre-trained Xception model for feature extraction, and image
        transformation steps.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=False,
            thresholds=[0.7, 0.8, 0.9],  # TIGHTER thresholds for higher face detection confidence
            min_face_size=40,             # Slightly larger minimum face to avoid tiny bad detections
            post_process=True,
            device=self.device
        )

        self.model = self._load_xception_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
            transforms.Normalize([0.5], [0.5])
        ])

    def _load_xception_model(self):
        """
        Loads a pre-trained Xception model using timm, removes the final classification
        layer, and returns the model in evaluation mode.
        """
        model = timm.create_model('xception', pretrained=True)
        model.fc = nn.Identity()
        model.eval().to(self.device)
        return model

    def detect_face(self, frame):
        """
        Detects a face in the input frame (BGR format from OpenCV) using MTCNN.

        Returns the cropped face if detected, otherwise None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = self.detector(rgb)
        return face

    def extract_feature(self, face_img):
        """
        Extracts feature embeddings from a face image using the Xception model.
        Accepts either a torch.Tensor or a NumPy array/PIL image.
        """
        if isinstance(face_img, torch.Tensor):
            tensor = face_img.unsqueeze(0).to(self.device)
        else:
            tensor = self.transform(face_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)
        return features.cpu().numpy().flatten()

    def process_frame(self, frame):
        """
        Processes a single video frame to detect a face and extract its features.

        Returns None if no face is detected, otherwise returns feature array.
        """
        face = self.detect_face(frame)
        if face is None:
            return None
        return self.extract_feature(face)
