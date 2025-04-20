# --- Improved Temporal Classifier (Same function names) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Temporal Consistency Module ---
def compute_framewise_temporal_scores(feature_dict):
    filenames = sorted(feature_dict.keys())
    vectors = [feature_dict[name] for name in filenames]

    frame_scores = {}
    frame_scores[filenames[0]] = 1.0

    for i in range(1, len(vectors)):
        sim = cosine_similarity([vectors[i - 1]], [vectors[i]])[0][0]
        frame_scores[filenames[i]] = sim

    return frame_scores

# --- Improved Classifier Model (added BatchNorm, extra layer, deeper network) ---
class Classifier(nn.Module):  # Keeping the same name 'Classifier'
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.4)

        self.fc4 = nn.Linear(64, 32)  # NEW layer
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.3)

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.fc4(x)))  # NEW layer in forward pass
        x = self.dropout4(x)

        return torch.sigmoid(self.out(x))

# --- Classification Pipeline ---
def classify_features(real_features, fake_features, input_dim=2048, save_to_file=True):
    model = Classifier(input_dim + 1)  # Still using the same class name
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # changed step_size from 50 -> 30
    criterion = nn.BCELoss()

    real_scores = compute_framewise_temporal_scores(real_features)
    fake_scores = compute_framewise_temporal_scores(fake_features)

    data = []
    for name, feat in real_features.items():
        temporal_score = real_scores.get(name, 1.0)
        augmented_feat = np.append(feat, temporal_score)
        data.append((name, augmented_feat, 1))
    for name, feat in fake_features.items():
        temporal_score = fake_scores.get(name, 1.0)
        augmented_feat = np.append(feat, temporal_score)
        data.append((name, augmented_feat, 0))

    names, X, y = zip(*data)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(
        X, y, names, test_size=0.3, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    print(f"\n🖼️ Total extracted feature samples: {len(names)} (real + fake combined)")  # Updated print

    model.train()
    num_epochs = 200  # increased from 100 -> 200
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        pred_labels = (preds > 0.5).int()

        acc = (pred_labels == y_test.int()).float().mean().item()
        print(f"\n🔍 Test Accuracy: {acc * 100:.2f}%")

        y_true = y_test.cpu().numpy()
        y_pred = pred_labels.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"\n📊 Confusion Matrix:\n{cm}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"📈 Recall:    {recall:.4f}")
        print(f"🧬 F1 Score:  {f1:.4f}")

        results = []
        for name, pred in zip(name_test, pred_labels):
            label = "Real" if pred.item() == 1 else "Fake"
            print(f"{name} -> {label}")
            results.append((name, label))

        if save_to_file:
            df = pd.DataFrame(results, columns=["Filename", "PredictedLabel"])
            df.to_csv("predictions.csv", index=False)
            print("\n📄 Saved predictions to predictions.csv")

    return model
