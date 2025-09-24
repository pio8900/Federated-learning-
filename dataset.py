import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("heart.csv")  # Ensure this file exists

# Separate features & labels
X = df.drop(columns=["output"]).values
y = df["output"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simulate 3 clients (split training data)
num_clients = 3
client_data = np.array_split(X_train, num_clients)
client_labels = np.array_split(y_train, num_clients)

# Save preprocessed data correctly using pickle
with open("data.pkl", "wb") as f:
    pickle.dump({
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "client_data": client_data, "client_labels": client_labels
    }, f)

print("✅ Data preprocessing complete. Data saved as 'data.pkl'")
