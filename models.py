from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression Model
def train_logistic_regression(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

# Support Vector Machine (SVM) Model
def train_svm(X, y):
    model = SVC(probability=True, kernel="linear")
    model.fit(X, y)
    return model

# Random Forest Model
def train_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
