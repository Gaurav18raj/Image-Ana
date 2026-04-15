# src/train_model.py
# src/train_model.py

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# IMPORTANT FIX
from preprocess import load_data

def train():

    print("Loading data...")

    # Load dataset
    X, y = load_data("data/train")

    print("Data loaded successfully!")
    print(f"Total samples: {len(X)}")

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")

    # Create model (SVM)
    model = SVC(kernel='linear')

    # Train model
    model.fit(X_train, y_train)

    print("Model training completed!")

    # Predict on test data
    y_pred = model.predict(X_test)

    # Check accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")

    # Save model
    joblib.dump(model, "model/model.pkl")

    print("Model saved in model/model.pkl")

if __name__ == "__main__":
    train()