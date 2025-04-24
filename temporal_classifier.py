# PyTorch + ML-related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# sklearn utilities for evaluation, preprocessing, and splitting
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# --- Temporal Similarity Calculation (for frame consistency) ---
def compute_framewise_temporal_scores(feature_dict):
    filenames = sorted(feature_dict.keys())  # ensure order is correct
    vectors = [feature_dict[name] for name in filenames]
    frame_scores = {filenames[0]: 1.0}  # first frame has max consistency by default

    # compute cosine similarity between each pair of consecutive feature vectors
    for i in range(1, len(vectors)):
        sim = cosine_similarity([vectors[i - 1]], [vectors[i]])[0][0]
        frame_scores[filenames[i]] = sim
    return frame_scores

# ----------------------------------------------------------------------------------------
# MLP Classifier Architecture (Fully Connected Neural Network)
#
# This class defines the architecture of the deep neural network used to classify whether
# a given input feature vector represents a real face or a fake one (i.e., deepfake).
#
# Input:
#   - Each input vector has 2049 dimensions:
#       - 2048 features from a pre-trained Xception model (facial embedding)
#       - 1 temporal consistency score calculated from frame similarity
#
# Architecture Overview:
#   - The model is a deep feedforward neural network (multi-layer perceptron)
#   - It follows a "funnel" structure: each layer reduces the dimensionality to focus
#     the network's attention and reduce overfitting
#   - Every layer uses Batch Normalization (for stable training) and Dropout (for regularization)
#   - Activation function used between layers is ReLU, and the output is passed through a Sigmoid
#     to produce a probability (0 = fake, 1 = real)
#
# Layer Sizes:
#   - Input: 2049
#   - Hidden Layers: 512 → 256 → 64 → 32
#   - Output: 1
#
# Why This Design:
#   - 512 is a good initial size to compress high-dimensional Xception features
#   - Halving each layer’s size gradually reduces feature space while still allowing
#     complex decision boundaries to be formed
#   - Dropout rates are higher in early layers to prevent overfitting on large features,
#     and reduced slightly toward the output to improve stability
#   - Final Sigmoid output is perfect for binary classification tasks like real vs. fake
#
# This model is compact, efficient, and deep enough to learn subtle patterns in both
# visual and temporal cues without being overly complex or slow to train.
# ----------------------------------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()

        # Input is 2049 (2048 from Xception + 1 temporal score)
        # First layer reduces dimensionality while still keeping lots of room for abstraction
        self.fc1 = nn.Linear(input_dim, 512)  # 2049 → 512: compress by 4x
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)  # Heavier dropout early to regularize big feature space

        # Second layer cuts size in half again, keeps abstraction strong but trims complexity
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)  # Same dropout to maintain regularization through depth

        # Smaller third layer brings it closer to decision boundaries
        self.fc3 = nn.Linear(256, 64)  # A more compact space for dense decision patterns
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.4)

        # Last hidden layer before output — small, efficient, less prone to overfitting
        self.fc4 = nn.Linear(64, 32)  # Final compression before decision
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.3)  # Slightly lower dropout to stabilize final output

        # Output layer → single value between 0 and 1 for binary classification
        self.out = nn.Linear(32, 1)  # 1 neuron = sigmoid probability (real vs fake)

    def forward(self, x):
        # Run through each FC → BN → ReLU → Dropout layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        return torch.sigmoid(self.out(x))  # sigmoid for binary classification


# -------------------------------------------------------------------------------------------------
# classify_features()
#
# This is the main function that handles training and evaluating the deepfake detection model.
# It takes in pre-extracted facial features (from Xception) and temporal scores for both
# real and fake images, prepares the data for training, builds the classifier, trains it,
# evaluates performance, and saves everything that matters (metrics, plots, model, etc).
#
# INPUTS:
#   - real_features: Dictionary of feature vectors for real face images
#   - fake_features: Dictionary of feature vectors for fake (deepfake) face images
#   - input_dim: Feature vector size (default is 2048 from Xception)
#   - save_to_file: Whether or not to save predictions and metrics
#   - load_checkpoint: If True, loads a pre-trained model checkpoint instead of training from scratch
#
# WHAT IT DOES:
#   - Appends a temporal consistency score to each feature vector (2048 → 2049)
#   - Combines real and fake features, standardizes them, and splits into train/test
#   - Trains a multi-layer MLP classifier over 200 epochs with dropout, batchnorm, and Adam optimizer
#   - Logs loss and accuracy every epoch
#   - Plots and saves training curves (loss + accuracy)
#   - Evaluates on the test set and computes all key metrics: Accuracy, Precision, Recall, F1, AUC-ROC
#   - Plots the AUC-ROC curve
#   - Saves evaluation results to `metrics_summary.csv` and model weights to `temporal_classifier_checkpoint.pth`
#
# OUTPUTS:
#   - Returns the trained model (or loaded model if checkpoint is used)
#   - Saves all relevant visualizations and logs to disk
#
# This is the central function of the entire classification system.
# -------------------------------------------------------------------------------------------------
def classify_features(real_features, fake_features, input_dim=2048, save_to_file=True, load_checkpoint=False):
    model = Classifier(input_dim + 1)  # +1 to account for the temporal score
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    # Print out the training config and hardware details
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only'}")
    print("🔧 Training Parameters:")
    print(f"   - Epochs: 200")
    print(f"   - Batch size: 64")
    print(f"   - Learning rate: 0.001")
    print(f"   - Optimizer: Adam")
    print(f"   - Loss: BCELoss")
    print(f"   - Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if load_checkpoint:
        model.load_state_dict(torch.load("temporal_classifier_checkpoint.pth"))
        print("🧠 Loaded model checkpoint.")

    # Add temporal scores to features
    real_scores = compute_framewise_temporal_scores(real_features)
    fake_scores = compute_framewise_temporal_scores(fake_features)

    data = []
    for name, feat in real_features.items():
        temporal_score = real_scores.get(name, 1.0)
        data.append((name, np.append(feat, temporal_score), 1))
    for name, feat in fake_features.items():
        temporal_score = fake_scores.get(name, 1.0)
        data.append((name, np.append(feat, temporal_score), 0))

    # Prepare data for training
    names, X, y = zip(*data)
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(
        X, y, names, test_size=0.3, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Wrap data in PyTorch DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    # Training loop
    model.train()
    num_epochs = 200
    losses, accuracies = [], []

    print("⏳ Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Tracks cumulative loss for the epoch
        correct = 0       # Counts correct predictions
        total = 0         # Total samples seen in this epoch

        # Loop through mini-batches
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()              # Clear previous gradients
            output = model(batch_X)            # Forward pass through the network
            loss = criterion(output, batch_y)  # Compute binary cross-entropy loss
            loss.backward()                    # Backpropagation to compute gradients
            optimizer.step()                   # Update model weights

            epoch_loss += loss.item()          # Accumulate batch loss

            # Convert probabilities to binary predictions (0 or 1)
            pred = (output > 0.5).int()
            correct += (pred == batch_y.int()).sum().item()  # Count correct predictions
            total += batch_y.size(0)            # Track total number of samples

        scheduler.step()  # Adjust learning rate if using a scheduler

        avg_loss = epoch_loss / len(train_loader)  # Compute average loss for this epoch
        accuracy = correct / total                 # Compute accuracy for this epoch

        losses.append(avg_loss)        # Save loss to plot later
        accuracies.append(accuracy)    # Save accuracy to plot later

        # Print progress for this epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


    duration = time.time() - start_time
    print(f"🕒 Total Training Time: {duration:.2f} seconds")

    # Save trained model
    torch.save(model.state_dict(), "temporal_classifier_checkpoint.pth")
    print("💾 Model checkpoint saved.")

    # Plot and save loss curve
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

    # Plot and save accuracy curve
    plt.figure()
    plt.plot(range(1, num_epochs + 1), [a * 100 for a in accuracies], label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy Curve")
    plt.grid(True)
    plt.savefig("accuracy_curve.png")
    plt.show()

    # --- Evaluation on Test Set ---
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        pred_labels = (preds > 0.5).int()

        y_true = y_test.cpu().numpy()
        y_pred = pred_labels.cpu().numpy()
        y_probs = preds.cpu().numpy()

        # Metrics
        acc = (pred_labels == y_test.int()).float().mean().item()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n📊 Confusion Matrix:\n{cm}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"📈 Recall:    {recall:.4f}")
        print(f"🧬 F1 Score:  {f1:.4f}")
        print(f"📐 AUC-ROC:   {auc:.4f}")
        print(f"✅ Test Accuracy: {acc * 100:.2f}%")

        # AUC-ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("AUC-ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("roc_curve.png")
        plt.show()

        # Save metrics to CSV
        metrics = {
            "Accuracy": [acc],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [f1],
            "AUC": [auc],
            "Train Time (s)": [duration],
        }
        pd.DataFrame(metrics).to_csv("metrics_summary.csv", index=False)
        print("📄 Metrics saved to metrics_summary.csv")

    return model
