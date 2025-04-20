# Deepfake Detection Pipeline Overview

This modular pipeline detects deepfakes by combining face detection, feature extraction using a pre-trained Xception model, classification via a neural network, and an analysis of temporal consistency between frames. Each module serves a clear and distinct purpose within the full detection process.
------------------------------------------------------------------------
Updated how the model works - 4/8/2025:
The pipeline begins by extracting frames from videos in two classes — real and fake — located in separate folders. From each video, about 10 frames are sampled evenly across the video’s length. During extraction, each frame is checked for sharpness and brightness to ensure only clear, good-quality frames are selected for further processing.

Once frames are extracted, the model uses MTCNN (Multi-task Cascaded Convolutional Networks) to perform face detection. MTCNN crops out only the face region from each frame, ignoring the background and surroundings. This ensures the model focuses purely on the face — where deepfake artifacts are most likely to appear. Each detected face is resized and normalized to a standard format (299×299) suitable for feature extraction.

For feature extraction, a pre-trained Xception model (with its classification head removed) is used. Each face is passed through Xception, which outputs a 2048-dimensional feature vector that captures high-level facial characteristics. A temporal consistency score (based on cosine similarity between consecutive frame features) is also computed and appended to each feature vector, enriching the input data.

Next, these features are fed into a deep Multi-Layer Perceptron (MLP) classifier. The MLP consists of multiple hidden layers (512 → 256 → 64 neurons) with Dropout applied after each layer to prevent overfitting. The model is trained using binary cross-entropy loss to classify faces as real or fake. To improve training dynamics, feature normalization is applied, and a learning rate scheduler (StepLR) gradually reduces the learning rate after 50 epochs to fine-tune the learning.

Finally, the model’s performance is evaluated using metrics such as accuracy, precision, recall, and F1 score. The results demonstrate how well the model can detect deepfakes based purely on facial artifacts, without being distracted by irrelevant background information.
------------------------------------------------------------------------


---

## 1. Face Feature Extraction (`face_features.py`)

Purpose:  
The `FaceFeatureExtractor` class handles both face detection and feature extraction.

- Face Detection:  
  Utilizes MTCNN (Multi-task Cascaded Convolutional Networks) from `facenet-pytorch` to locate and align faces in input frames.

- Feature Extraction:  
  Loads a pre-trained Xception model via `timm`, removes the final classification layer, and uses the model to output 2048-dimensional feature embeddings for each detected face.

- How It Fits In:  
  This class is used to transform input images into consistent feature vectors, which are essential for classification and temporal analysis.

---

## 2. Image Processing and Pipeline Execution (`main.py`)

Purpose:  
Coordinates the full deepfake detection pipeline, handling:
- Data loading
- Face feature extraction
- Temporal score computation
- Feature classification

Main Steps:
- Loads `.jpg` images from `ExtractedFrames/FakeExtracted` and `ExtractedFrames/RealExtracted`.
- Uses `FaceFeatureExtractor` to extract features from each image.
- Calls `classify_features()` to evaluate and classify real vs. fake features.

How It Fits In:  
Acts as the central orchestrator of the pipeline. It ties together feature extraction, classification, and reporting.

---

## 3. Classification Layer (`temporal_classifier.py`)

Purpose:  
Trains a binary classifier to distinguish between real and fake images based on their feature embeddings and temporal coherence.

Key Details:
- Implements a simple Multi-Layer Perceptron (MLP) using PyTorch:
  - Input → 256 → 64 → 1 with ReLU activations and Sigmoid output
- Uses BCELoss (Binary Cross-Entropy) and the Adam optimizer
- Training data includes an additional temporal consistency score for each image
- Automatically splits data into training/testing sets and reports metrics

Outputs:
- Accuracy, precision, recall, F1 score
- Confusion matrix
- CSV file (`predictions.csv`) containing predicted labels for each test image

How It Fits In:  
This is the core decision-making module. It determines whether each image is likely real or fake based on visual features and temporal stability.

---

## 4. Temporal Consistency Check (`temporal_classifier.py`)

Purpose:  
Computes a temporal similarity score for each frame based on cosine similarity between consecutive feature vectors.

Key Function:
- `compute_framewise_temporal_scores()`:  
  Calculates the similarity between each frame and its previous one, assigning a value between 0 and 1.

Why It Matters:  
Real videos typically exhibit smooth feature transitions across frames. Deepfakes often show temporal jitter or inconsistencies that can be caught through this metric.

How It Fits In:  
These temporal scores are used as an additional feature during classification, improving the model’s ability to detect inconsistencies across time.

---

## End-to-End Workflow

1. `main.py` loads images and initializes the feature extractor
2. Each image is processed through `FaceFeatureExtractor` to get 2048-dim feature vectors
3. `compute_framewise_temporal_scores()` assigns a temporal score to each image
4. `classify_features()` trains a classifier using both feature vectors and their temporal scores
5. Results are evaluated and saved for further analysis

---

## Citations & Libraries

1. MTCNN for Face Detection  
   Library: `facenet-pytorch`  
   GitHub: https://github.com/timesler/facenet-pytorch  
   Based on: Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks", IEEE Signal Processing Letters, 2016

2. Xception Model for Feature Extraction  
   Library: `timm` (PyTorch Image Models by Ross Wightman)  
   GitHub: https://github.com/rwightman/pytorch-image-models  
   Original paper: "Xception: Deep Learning with Depthwise Separable Convolutions" by François Chollet (2017)

3. Classification Layer  
   Simple Multi-Layer Perceptron (MLP), inspired by PyTorch official tutorials:  
   https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html  
   Adapted for binary classification using sigmoid activation and BCELoss

4. Temporal Consistency  
   Based on cosine similarity of features across sequential frames  
   Inspired by temporal analysis in FakeCatcher  
   Scikit-learn cosine similarity:  
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
