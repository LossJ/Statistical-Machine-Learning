import numpy as np


class NumpyMinMaxScaler:
    """Min Max Normalization Scaler with Numpy

    Attributes:
        min_: The min vector of features, with (feature_count, ) shape.
        max_: The max vector of features, with (feature_count, ) shape.
    """
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X_train):
        """Fit the scaler from X_train data.

        Args:
            X_train: The train data of features, with (sample_count, feature_count) shape.
        """
        self.max_ = np.max(X_train, axis=0)
        self.min_ = np.min(X_train, axis=0)

    def transform(self, X):
        """Transform X data.

        Args:
            X: The data of features, with (sample_count, feature_count) shape.

        Returns:
            A new matrix with transformation, with the same shape of old X.
        """
        return (X - self.min_) / (self.max_ - self.min_ + 0.00001)

    def fit_transform(self, X_train):
        """Fit and Transform X_train data.

        Args:
            X_train: The train data of features, with (sample_count, feature_count) shape.

        Returns:
            A new matrix with transformation, with the same shape of old X_train.
        """
        self.fit(X_train)
        return self.transform(X_train)