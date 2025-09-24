import os

print("\n🔄 Running Federated Learning...")

# Step 1: Load Data
os.system("python3 dataset.py")

# Step 2: Train Local Models & Aggregate
os.system("python3 train.py")

# Step 3: Evaluate Global Model
os.system("python3 evaluate.py")

print("\n🎉 Federated Learning Completed!")
