from collections import Counter

from .base import NumpyKNNBase, TFKNNBase, TorchKNNBase


import numpy as np
import tensorflow as tf
import torch


class NumpyKNNClassifier(NumpyKNNBase):
    """kNN Classifier with numpy, explicitly inherits from NumpyKNNBase already.

    Attributes:
        n_neighbors: A int number, number of neighbors.
        _metric: A method object, choose from {_manhattan_distance, _euclidean_distance, _chebyshev_distance}.
        _X_train: feature data for training. A np.ndarray matrix of (n_samples, n_features) shape,
            data type must be continuous value type.
        _y_train: label data for training. A np.ndarray array of (n_samples, ) shape,
            data type must be discrete value.
    """

    def score(self, X_test, y_test):
        """Use test dataset to evaluate the trained model.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape.
            y_test: A np.ndarray array of (n_samples, ) shape. data type must be
                discrete value.
        Returns:
            return accuracy, a float number. accuracy = correct_count / y_test.shape[0]
        """
        X_test, y_test = self._score_validation(X_test, y_test)

        y_pred = self.predict(X_test)
        correct_count = np.sum(y_pred == y_test)
        accuracy = correct_count / y_test.shape[0]
        return accuracy

    def _predict_sample(self, sample):
        k_labels = self._find_k_labels(sample)
        pred = Counter(k_labels).most_common(1)[0][0]
        return pred


class TFKNNClassifier(TFKNNBase):
    """kNN Classifier with TensorFlow, explicitly inherits from TFKNNBase already.

    Attributes:
        n_neighbors: A int number, number of neighbors.
        _metric: A method object, choose from {_manhattan_distance, _euclidean_distance, _chebyshev_distance}.
        _X_train: feature data for training. A tf.Tensor matrix of (n_samples, n_features) shape,
            data type must be continuous value type.
        _y_train: label data for training. A tf.Tensor array of (n_samples, ) shape,
            data type must be discrete value.
    """

    def score(self, X_test, y_test):
        """Use test dataset to evaluate the trained model.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape.
            y_test: A np.ndarray array of (n_samples, ) shape. data type must be
                discrete value.
        Returns:
            return accuracy, a float number. accuracy = correct_count / y_test.shape[0]
        """
        X_test, y_test = self._score_validation(X_test, y_test)

        y_pred = self.predict(X_test)
        correct_count = tf.reduce_sum(tf.cast(y_pred == y_test, dtype=tf.int32))
        accuracy = correct_count / y_test.shape[0]
        return accuracy.numpy()

    def _predict_sample(self, sample):
        k_labels = self._find_k_labels(sample)
        pred = Counter(k_labels.numpy()).most_common(1)[0][0]
        return pred


class TorchKNNClassifier(TorchKNNBase):
    """kNN Classifier with PyTorch, explicitly inherits from TorchKNNBase already.

    Attributes:
        n_neighbors: A int number, number of neighbors.
        _metric: A method object, choose from {_manhattan_distance, _euclidean_distance, _chebyshev_distance}.
        _X_train: feature data for training. A torch.Tensor matrix of (n_samples, n_features) shape,
            data type must be continuous value type.
        _y_train: label data for training. A torch.Tensor array of (n_samples, ) shape,
            data type must be discrete value.
    """

    def score(self, X_test, y_test):
        """Use test dataset to evaluate the trained model.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape.
            y_test: A np.ndarray array of (n_samples, ) shape. data type must be
                discrete value.
        Returns:
            return accuracy, a float number. accuracy = correct_count / y_test.shape[0]
        """
        X_test, y_test = self._score_validation(X_test, y_test)

        y_pred = torch.tensor(self.predict(X_test))
        correct_count = torch.sum(y_pred == y_test)
        accuracy = correct_count.item() / y_test.shape[0]
        return accuracy

    def _predict_sample(self, sample):
        k_labels = self._find_k_labels(sample)
        pred = Counter(k_labels.numpy()).most_common(1)[0][0]
        return pred
