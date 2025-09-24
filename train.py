import pickle
import numpy as np
from models import train_logistic_regression, train_svm, train_random_forest

# Load preprocessed data
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

client_data = data["client_data"]
client_labels = data["client_labels"]

# Train models on each client
def train_clients():
    local_models = []
    for i in range(len(client_data)):
        print(f"🔹 Training Client {i+1}...")

        # Train different models
        model1 = train_logistic_regression(client_data[i], client_labels[i])
        model2 = train_svm(client_data[i], client_labels[i])
        model3 = train_random_forest(client_data[i], client_labels[i])

        local_models.append((model1, model2, model3))

    return local_models

# ✅ New Federated Averaging Implementation (fix for RandomForest)
def federated_averaging(models):
    num_clients = len(models)

    # Aggregate Logistic Regression coefficients
    coef_logistic = sum(m[0].coef_ for m in models) / num_clients
    intercept_logistic = sum(m[0].intercept_ for m in models) / num_clients

    # ✅ Train a new SVM model using aggregated data
    X_train = np.vstack(client_data)  # Combine all client data
    y_train = np.hstack(client_labels)
    global_svm = train_svm(X_train, y_train)  # Train a new SVM model

    # ✅ Train a new Random Forest model using aggregated data
    global_rf = train_random_forest(X_train, y_train)

    # Train a new global logistic regression model
    global_logistic = train_logistic_regression(X_train, y_train)
    global_logistic.coef_ = coef_logistic
    global_logistic.intercept_ = intercept_logistic

    return global_logistic, global_svm, global_rf

# Run Federated Training
local_models = train_clients()
global_models = federated_averaging(local_models)

# Save global models correctly
with open("global_models.pkl", "wb") as f:
    pickle.dump(global_models, f)

print("✅ Federated training complete. Global models saved as 'global_models.pkl'")
