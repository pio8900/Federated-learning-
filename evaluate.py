import pickle
from sklearn.metrics import accuracy_score

# Load test data
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

X_test, y_test = data["X_test"], data["y_test"]

# Load trained global models
try:
    with open("global_models.pkl", "rb") as f:
        global_models = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: Global model file not found. Run `train.py` first.")
    exit()

global_logistic, global_svm, global_rf = global_models

# Evaluate each model
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {acc:.4f}")

evaluate_model(global_logistic, X_test, y_test, "Logistic Regression")
evaluate_model(global_svm, X_test, y_test, "SVM")
evaluate_model(global_rf, X_test, y_test, "Random Forest")
