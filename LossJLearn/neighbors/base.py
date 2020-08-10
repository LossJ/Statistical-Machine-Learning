from collections import Iterable


import numpy as np
import tensorflow as tf
import torch


class NumpyKNNBase:
    """KNN basic class with Numpy.

    Attributes:
        n_neighbors: A int number, number of neighbors.
        _metric: A method object, choose from {_manhattan_distance, _euclidean_distance, _chebyshev_distance}.
        _X_train: feature data for training. A np.ndarray matrix of (sample_lenght, feature_lenght) shape,
            data type must be continuous value type.
        _y_train: label data for training. A np.ndarray array of (sample_lenght, ) shape,
            data type must be discrete value.
    """

    def __init__(self, n_neighbors=5, metric="euclidean"):
        """Init method.

        Args:
            n_neighbors: int, optional (default = 5), the integer must greater then 0.
                Number of neighbors to use by default for :meth:`kneighbors` queries.
            metric: {"manhattan", "euclidean", "chebyshev"}, optional, default 'euclidean'.

        Raises:
            ValueError: metric value is out of options.
            AssertionError: n_neighbors value is not a integer or n_neighbors > 0.
        """
        assert isinstance(n_neighbors, int) and n_neighbors > 0
        self.n_neighbors = n_neighbors
        if metric == "manhattan":
            self._metric = self._manhattan_distance
        elif metric == "euclidean":
            self._metric = self._euclidean_distance
        elif metric == "chebyshev":
            self._metric = self._chebyshev_distance
        else:
            raise ValueError(f'No such metric as {metric}, please option from: {"manhattan", "euclidean", "chebyshev"}')
        self._X_train, self._y_train = [None] * 2

    def __new__(cls):
        raise Exception("Can't instantiate an object from NumpyKNNBase class! ")

    def fit(self, X_train, y_train):
        """method for training model.

        Args:
            X_train: A np.ndarray matrix of (sample_lenght, feature_lenght) shape, data type must be continuous value type.
            y_train: A np.ndarray array of (sample_lenght, ) shape.

        Raises:
            AssertionError: X_train value or y_train value with a mismatched shape.
        """
        assert isinstance(X_train, Iterable) and isinstance(y_train, Iterable)
        assert len(X_train) == len(y_train)
        self._X_train = X_train if isinstance(X_train, np.ndarray) else np.array(X_train)
        self._y_train = y_train if isinstance(y_train, np.ndarray) else np.array(y_train)

    def predict(self, X_test):
        """predict test data.

        Args:
            X_test: A np.ndarray matrix of (sample_lenght, feature_lenght) shape,
                or a np.ndarray array of (feature_lenght, ) shape.

        Returns:
            A list for samples predictions or a single prediction.

        Raises:
            ValueError: X_test value with a mismatched shape.
        """
        assert isinstance(X_test, Iterable)
        X_test = X_test if isinstance(X_test, np.ndarray) else np.array(X_test)

        if X_test.shape == (self._X_train.shape[1],):
            y_pred = self._predict_sample(X_test)
        elif X_test.shape[1] == self._X_train.shape[1]:
            y_pred = []
            for sample in X_test:
                y_pred.append(self._predict_sample(sample))
        else:
            raise ValueError("Mismatched shape for X_test")
        return y_pred

    def _manhattan_distance(self, x):
        return np.sum(np.abs(self._X_train - x), axis=1)

    def _euclidean_distance(self, x):
        return np.sqrt(np.sum(np.square(self._X_train - x), axis=1))

    def _chebyshev_distance(self, x):
        return np.max(np.abs(self._X_train - x), axis=1)

    def _find_k_labels(self, sample):
        distance = self._metric(sample)
        sorted_idx = np.argsort(distance)
        k_labels = self._y_train[sorted_idx[:self.n_neighbors]]
        return k_labels

    def _predict_sample(self, sample):
        raise Exception("Can call predict method for NumpyKNNBase object! ")

    def _score_validation(self, X_test, y_test):
        assert isinstance(X_test, Iterable) and isinstance(y_test, Iterable)
        assert len(X_test) == len(y_test)
        X_test = X_test if isinstance(X_test, np.ndarray) else np.array(X_test)
        y_test = y_test if isinstance(y_test, np.ndarray) else np.array(y_test)
        return X_test, y_test


class TFKNNBase:
    """KNN basic class with TensorFlow.

    Attributes:
        n_neighbors: A int number, number of neighbors.
        _metric: A method object, choose from {_manhattan_distance, _euclidean_distance, _chebyshev_distance}.
        _X_train: feature data for training. A tf.Tensor matrix of (sample_lenght, feature_lenght) shape,
            data type must be continuous value type.
        _y_train: label data for training. A tf.Tensor array of (sample_lenght, ) shape,
            data type must be discrete value.
    """

    def __new__(cls):
        raise Exception("Can't instantiate an object from TFKNNBase! ")

    def __init__(self, n_neighbors=5, metric="euclidean"):
        """Init method.

        Args:
            n_neighbors: int, optional (default = 5), the integer must greater then 0.
                Number of neighbors to use by default for :meth:`kneighbors` queries.
            metric: {"manhattan", "euclidean", "chebyshev"}, optional, default 'euclidean'.

        Raises:
            ValueError: metric value is out of options.
            AssertionError: n_neighbors value is not a integer or n_neighbors > 0.
        """
        assert isinstance(n_neighbors, int) and n_neighbors > 0
        self.n_neighbors = n_neighbors
        if metric == "manhattan":
            self._metric = self._manhattan_distance
        elif metric == "euclidean":
            self._metric = self._euclidean_distance
        elif metric == "chebyshev":
            self._metric = self._chebyshev_distance
        else:
            raise ValueError(f'No such metric as {metric}, please option from: {"manhattan", "euclidean", "chebyshev"}')
        self._X_train, self._y_train = [None] * 2

    def fit(self, X_train, y_train):
        """method for training model.

        Args:
            X_train: A matrix of (sample_lenght, feature_lenght) shape, data type must be continuous value type.
            y_train: A array of (sample_lenght, ) shape.

        Raises:
            AssertionError: X_train value or y_train value with a mismatched shape.
        """
        assert isinstance(X_train, Iterable) and isinstance(y_train, Iterable)
        assert len(X_train) == len(y_train)
        self._X_train = tf.convert_to_tensor(X_train, dtype=tf.dtypes.float32)
        self._y_train = y_train if isinstance(y_train, tf.Tensor) else tf.convert_to_tensor(y_train)

    def predict(self, X_test):
        """predict test data.

        Args:
            X_test: A np.ndarray matrix of (sample_lenght, feature_lenght) shape,
                or a np.ndarray array of (feature_lenght, ) shape.

        Returns:
            A list for samples predictions or a single prediction.

        Raises:
            ValueError: X_test value with a mismatched shape.
        """
        assert isinstance(X_test, Iterable)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.dtypes.float32)

        if X_test.shape == (self._X_train.shape[1],):
            y_pred = self._predict_sample(X_test)
        elif X_test.shape[1] == self._X_train.shape[1]:
            y_pred = []
            for sample in X_test:
                y_pred.append(self._predict_sample(sample))
        else:
            raise ValueError("Mismatched shape for X_test")
        return y_pred

    def _manhattan_distance(self, x):
        return tf.reduce_sum(tf.abs(self._X_train - x), axis=1)

    def _euclidean_distance(self, x):
        return tf.sqrt(tf.reduce_sum(tf.square(self._X_train - x), axis=1))

    def _chebyshev_distance(self, x):
        return tf.reduce_max(tf.abs(self._X_train - x), axis=1)

    def _find_k_labels(self, sample):
        distance = self._metric(sample)
        sorted_idx = tf.argsort(distance)
        k_labels = tf.gather(self._y_train, indices=sorted_idx[:self.n_neighbors])
        return k_labels

    def _predict_sample(self, sample):
        raise Exception("Can call predict method for NumpyKNNBase object! ")

    def _score_validation(self, X_test, y_test):
        assert isinstance(X_test, Iterable) and isinstance(y_test, Iterable)
        assert len(X_test) == len(y_test)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.dtypes.float32)
        y_test = y_test if isinstance(y_test, tf.Tensor) else tf.convert_to_tensor(y_test)
        return X_test, y_test


class TorchKNNBase:
    """KNN basic class with PyTorch.

    Attributes:
        n_neighbors: A int number, number of neighbors.
        _metric: A method object, choose from {_manhattan_distance, _euclidean_distance, _chebyshev_distance}.
        _X_train: feature data for training. A tf.Tensor matrix of (sample_lenght, feature_lenght) shape,
            data type must be continuous value type.
        _y_train: label data for training. A tf.Tensor array of (sample_lenght, ) shape,
            data type must be discrete value.
    """

    def __new__(cls):
        raise Exception("Can't instantiate an object from TorchKNNBase! ")

    def __init__(self, n_neighbors=5, metric="euclidean"):
        """Init method.

        Args:
            n_neighbors: int, optional (default = 5), the integer must greater then 0.
                Number of neighbors to use by default for :meth:`kneighbors` queries.
            metric: {"manhattan", "euclidean", "chebyshev"}, optional, default 'euclidean'.

        Raises:
            ValueError: metric value is out of options.
            AssertionError: n_neighbors value is not a integer or n_neighbors > 0.
        """
        assert isinstance(n_neighbors, int) and n_neighbors > 0
        self.n_neighbors = n_neighbors
        if metric == "manhattan":
            self._metric = self._manhattan_distance
        elif metric == "euclidean":
            self._metric = self._euclidean_distance
        elif metric == "chebyshev":
            self._metric = self._chebyshev_distance
        else:
            raise ValueError(f'No such metric as {metric}, please option from: {"manhattan", "euclidean", "chebyshev"}')
        self._X_train, self._y_train = [None] * 2

    def fit(self, X_train, y_train):
        """method for training model.

        Args:
            X_train: A matrix of (sample_lenght, feature_lenght) shape, data type must be continuous value type.
            y_train: A array of (sample_lenght, ) shape.

        Raises:
            AssertionError: X_train value or y_train value with a mismatched shape.
        """
        assert isinstance(X_train, Iterable) and isinstance(y_train, Iterable)
        assert len(X_train) == len(y_train)
        self._X_train = torch.tensor(X_train, dtype=torch.float32)
        self._y_train = y_train if isinstance(y_train, torch.Tensor) else torch.tensor(y_train)

    def predict(self, X_test):
        """predict test data.

        Args:
            X_test: A np.ndarray matrix of (sample_lenght, feature_lenght) shape,
                or a np.ndarray array of (feature_lenght, ) shape.

        Returns:
            A list for samples predictions or a single prediction.

        Raises:
            ValueError: X_test value with a mismatched shape.
        """
        assert isinstance(X_test, Iterable)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        if X_test.shape == (self._X_train.shape[1],):
            y_pred = self._predict_sample(X_test)
        elif X_test.shape[1] == self._X_train.shape[1]:
            y_pred = []
            for sample in X_test:
                y_pred.append(self._predict_sample(sample))
        else:
            raise ValueError("Mismatched shape for X_test")
        return y_pred

    def _manhattan_distance(self, x):
        return torch.sum(torch.abs(self._X_train - x), dim=1)

    def _euclidean_distance(self, x):
        return torch.sqrt(torch.sum(torch.square(self._X_train - x), dim=1))

    def _chebyshev_distance(self, x):
        return torch.max(torch.abs(self._X_train - x), dim=1)

    def _find_k_labels(self, sample):
        distance = self._metric(sample)
        _, k_nearest_index = torch.topk(distance, self.n_neighbors, largest=False)
        k_labels = self._y_train[k_nearest_index]
        return k_labels

    def _predict_sample(self, sample):
        raise Exception("Can call predict method for NumpyKNNBase object! ")

    def _score_validation(self, X_test, y_test):
        assert isinstance(X_test, Iterable) and isinstance(y_test, Iterable)
        assert len(X_test) == len(y_test)
        y_test = y_test if isinstance(y_test, torch.Tensor) else torch.tensor(y_test)
        return X_test, y_test
