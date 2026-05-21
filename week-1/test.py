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
# print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")


# standardization
mean_train = X_train.mean()
std_train = X_train.std(ddof=0)

X_train_s = (X_train - mean_train) / std_train
X_test_s = (X_test - mean_train) / std_train

# print("Mean of standardized training features:")
# print(X_train_s.mean())

# print("\nStandard deviation of standardized training features:")
# print(X_train_s.std(ddof=0))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """A numerically stable sigmoid function."""
    # 1 / 1 + e^(-z)
    z = np.asarray(z)
    z = np.clip(z, -500, 500) # limit the numbers within -500 to 500
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Compute the mean binary cross-entropy loss.

    Take care in your implementation to ensure that the cross entropy is always positive,
    and that it stays stable for very small probabilities (y_hat \approx 0).
    """
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    
    #avoid log(0)
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1-eps)
    
    loss = -np.mean(y * np.log(y_hat) + (1 -  y) * np.log(1-y_hat))
    return loss

def logistic_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float, list[float]]:
    """Train logistic regression via gradient descent.

    Returns (weights, bias, loss_history).
    """
    # steps: for every feature sample - calculate z -  -  - find dw & db - update w & b untill loss changes no more
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    loss_hist = []

    for iter in range(max_iter):
        
        #calculate z
        z = X @ w + b #alternative: np.dot(X, W)

        #get probability
        y_hat = sigmoid(z)
        
        #calculate loss
        loss = binary_cross_entropy(y, y_hat)
        loss_hist.append(loss)

        #calculate dw
        error = y_hat - y
        # dw = (X.T @ (y_hat - y)) / n_samples
        dw = np.zeros(n_features)
        for j in range(n_features):
            total = 0.0
    
            for i in range(n_samples):
                total += X[i, j] * error[i]
            
            dw[j] = total / n_samples

        db = np.mean(y_hat - y)

        #update w and b
        for j in range(n_features):
            w[j] -= lr * dw[j]
        
        b -= lr * db

        #break if loss change too little
        if iter > 0 and abs(loss_hist[-1] - loss_hist[-2]) < tol:
            break
        
    return w, b, loss_hist



# convert python array to NumPy array
X_train_np = X_train_s.to_numpy()
X_test_np = X_test_s.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Train our model on the standardized features
w, b, loss_history = logistic_regression_gd(X_train_np, y_train_np, lr=0.1, max_iter=1000, tol=1e-8)

print(f"Final loss: {loss_history[-1]:.6f}")
print(f"Iterations: {len(loss_history)}")
print(f"\nLearned weights:")
for name, weight in zip(FEATURES, w):
    print(f"  {name:>30s}: {weight:+.4f}")
print(f"  {'bias':>30s}: {b:+.4f}")

def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict class labels (0 or 1)."""
    X = np.asarray(X)
    z = X @ w + b
    y_hat = sigmoid(z)

    # round to 0/1 with threshold 0.5
    y_pred = (y_hat >= 0.5).astype(int)
    return y_pred

# Evaluate on train and test sets
y_pred_train = predict(X_train_np, w, b)
y_pred_test = predict(X_test_np, w, b)

print("Train predictions shape:", y_pred_train.shape)
print("Test predictions shape:", y_pred_test.shape)

print("First 10 train predictions:", y_pred_train[:10])
print("First 10 test predictions:", y_pred_test[:10])

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return np.mean(y_true == y_pred)


print(f"Our model — Train accuracy: {accuracy(y_train_np, y_pred_train):.4f}")
print(f"Our model — Test accuracy:  {accuracy(y_test_np, y_pred_test):.4f}")