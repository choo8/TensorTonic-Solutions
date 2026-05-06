import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X, y = np.array(X), np.array(y)
    N, D = X.shape[0], X.shape[1]
    w, b = np.random.rand(D), np.random.rand()
    for _ in range(steps):
        y_hat = _sigmoid(np.array(X) @ w + b)
        dw = (1 / N) * (X.T @ (y_hat - y))
        db = (1 / N) * np.sum(y_hat - y)

        w -= lr * dw
        b -= lr * db

    return (w, b)