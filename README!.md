Deepfake Detection Pipeline Overview
  This modular pipeline detects deepfakes by combining face detection, deep feature extraction using a pre-trained Xception model, classification through a neural network, and analysis of temporal consistency across frames. Each module is independently structured to contribute to the accuracy and robustness of deepfake classification.

Updated Workflow — 4/23/2025:
  The pipeline starts by sampling frames from real and fake video folders. Each video yields around 10–25 frames, filtered for clarity (not blurry) and brightness. Selected frames are resized and padded to a standard 750×750 format for consistency.

  Face detection is performed using MTCNN, which extracts and aligns facial regions only — discarding the background. Each detected face is resized to 299×299 and normalized.

  For feature extraction, a pre-trained Xception model is used (classification head removed). The resulting 2048-dimensional facial features are further enriched with a temporal consistency score, computed using cosine similarity between consecutive frame features.

  These vectors are passed into a deep MLP classifier with the following architecture:

    Layers: 2049 → 512 → 256 → 64 → 32 → 1

    Components: BatchNorm, Dropout, ReLU, Sigmoid

    Optimization: Adam + BCELoss + StepLR scheduler

  The model is trained using mini-batches, and outputs are evaluated using:

    Accuracy, Precision, Recall, F1 Score

    AUC-ROC score and curve

    Confusion matrix

    Epoch-wise Loss & Accuracy plots

  Training metadata such as duration, parameter count, and hardware summary (GPU/CPU) are logged.

  A demo mode is also included, where previously extracted features are loaded from .pkl files, and a saved model checkpoint is used to skip retraining — enabling instant performance evaluation.

1. Face Feature Extraction (face_features.py)
Purpose:
Detects faces in input images and extracts 2048-dimensional embeddings.

Uses MTCNN to detect and crop aligned faces

Passes faces into Xception (from timm) with classification head removed

Outputs a compact feature vector representing high-level facial traits

2. Image Processing and Pipeline Execution (main.py)
Purpose:
Controls the full pipeline including preprocessing, feature extraction, training, and saving of intermediate outputs.

Loads .jpg images from real/fake frame folders

Runs FaceFeatureExtractor to get feature vectors

Saves those vectors to .pkl files for use in demo.py

Calls classify_features() to train and evaluate the MLP classifier

Logs CPU/GPU/RAM utilization and plots system performance

3. Classification Layer (temporal_classifier.py)
Purpose:
Binary classifier that predicts if a given feature vector (face + temporal score) is real or fake.

Multi-layer perceptron (MLP) with dropout, batch norm, ReLU

Input: 2048 (Xception) + 1 (temporal score) = 2049

Output: Sigmoid score (probability of being real)

Training: 200 epochs, BCELoss, Adam, StepLR

Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC

Saves: model checkpoint, performance plots, CSV metrics

4. Temporal Consistency Check (temporal_classifier.py)
Purpose:
Computes a similarity score between adjacent frame features to detect inconsistency.

Function: compute_framewise_temporal_scores()

Compares each frame’s feature to the previous one via cosine similarity

Used to capture jitter or instability common in deepfakes

5. Demo Mode (demo.py)
Purpose:
Provides a fast, checkpoint-based classification pipeline for demonstration.

Loads pre-saved real_features.pkl and fake_features.pkl

Loads trained model checkpoint from disk

Skips image processing and training

Evaluates instantly using the latest saved model

End-to-End Workflow Summary
  main.py loads images and initializes feature extractor

  FaceFeatureExtractor generates Xception features

  compute_framewise_temporal_scores() computes similarity scores

  classify_features() trains a deep MLP on features + temporal scores

  Results are saved and plotted

  demo.py runs inference instantly using cached features and model

Citations & Libraries
  1. Face Detection – MTCNN
  Library: facenet-pytorch

  Paper: Zhang et al. (2016). "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks", IEEE Signal Processing Letters. Link

  2. Feature Extraction – Xception
  Library: timm

  Paper: François Chollet (2017). "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR. Link

  3. MLP Classifier – PyTorch
  Framework: PyTorch

  Reference: PyTorch Neural Networks Tutorial

  Textbook Reference (Optional):

  Goodfellow et al. (2016). "Deep Learning", MIT Press — Chapter 6 covers feedforward neural networks and regularization via dropout. Book

  4. Temporal Consistency – Cosine Similarity
  Concept Origin: Inspired by temporal artifacts discussed in:

  Matern et al. (2019). "Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations", IEEE WIFS. Link

  Ciftci et al. (2020). "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals", IEEE TBIOM. Link

  Library: scikit-learn

  Metric Basis: Cosine similarity is a standard method for measuring angular distance between feature vectors in embedding space.

  5. Loss Function – Binary Cross Entropy
  Function Used: torch.nn.BCELoss

  Conceptual Reference:

  Bishop, C. M. (2006). "Pattern Recognition and Machine Learning", Springer — Covers probabilistic binary classification with sigmoid + BCE.

  6. Model Evaluation – AUC, Precision, F1
  Library: sklearn.metrics

  Evaluation Best Practices Reference:

  Saito & Rehmsmeier (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets", PLoS ONE. Link



