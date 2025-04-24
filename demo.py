import pickle
from temporal_classifier import classify_features

# Load pre-extracted features
with open("real_features.pkl", "rb") as f:
    real_features = pickle.load(f)
with open("fake_features.pkl", "rb") as f:
    fake_features = pickle.load(f)

# Run demo using loaded features and trained model
print("🚀 Running demo using saved features and model checkpoint...")
classify_features(real_features, fake_features, load_checkpoint=True)
