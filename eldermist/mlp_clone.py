import numpy as np

class TinyMLP:
    """
    One-hidden-layer neural classifier implemented only with NumPy.

    This is intentionally educational, not optimized.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.15, size=(in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.15, size=(hidden_dim, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)  # ReLU
        logits = h @ self.W2 + self.b2
        return z1, h, logits

    @staticmethod
    def softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(shifted)
        return ex / ex.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        _, _, logits = self.forward(X)
        return self.softmax(logits)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def train_step(self, X, y, lr=0.03):
        """
        Cross-entropy + manual backpropagation.
        """
        n = len(X)
        z1, h, logits = self.forward(X)
        probs = self.softmax(logits)

        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()

        # dL/dlogits for softmax + cross entropy
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        # Backprop through second linear layer
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T

        # Backprop through ReLU
        dz1 = dh * (z1 > 0)

        # Backprop through first linear layer
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        # THIS is the weight update.
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return float(loss)
