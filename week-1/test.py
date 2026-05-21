from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "titanic.csv"
data = pd.read_csv(data_path)
data.head()


FEATURES = ["sex", "age", "family_size", "fare", "1st_class", "2nd_class", "3rd_class"]
TARGET = "survived"



features = data[FEATURES]
target = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

mean_train = X_train.mean()
std_train = X_train.std(ddof=0)

X_train_s = (X_train - mean_train) / std_train
X_test_s = (X_test - mean_train) / std_train

print("Mean of standardized training features:")
print(X_train_s.mean())

print("\nStandard deviation of standardized training features:")
print(X_train_s.std(ddof=0))