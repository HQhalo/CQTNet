import numpy as np
# other
def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)
