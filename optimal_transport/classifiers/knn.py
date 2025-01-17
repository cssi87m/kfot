import numpy as np


class KNN:
    def __init__(self, n_clusters: int):
        self.k = n_clusters

    def fit(
        self, X: np.ndarray, 
        y: np.ndarray,
    ):
        self.X = X
        self.y = y

    def predict(
        self, X: np.ndarray
    ) -> np.ndarray:
        """
        K-Nearest Neighbors algorithm for classification.

        Parameters:
        - X_train: Training data features (numpy array).
        - y_train: Training data labels (numpy array).
        - X_test: Test data features (numpy array).
        - k: Number of neighbors to consider (default is 3).

        Returns:
        - y_pred: Predicted labels for test data.
        """
        assert self.X.shape[1] == X.shape[1], "Input data should have the same dimensions as the training data."
        
        distances = np.linalg.norm(self.X[:, np.newaxis, :] - X, axis=2)
        k_neighbors_indices = np.argsort(distances, axis=0)[:self.k]
        k_neighbors_labels = self.y[k_neighbors_indices]
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=np.max(self.y.astype("int64"))+1), axis=0, arr=k_neighbors_labels.astype("int64"))
        y_pred = np.argmax(counts, axis=0)

        return y_pred
    
def knn(n_clusters: int) -> KNN:
    return KNN(n_clusters=n_clusters)