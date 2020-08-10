from .base import NumpyKNNBase, TFKNNBase, TorchKNNBase


import numpy as np
import tensorflow as tf
import torch


class NumpyKNNRegressor(NumpyKNNBase):
    """kNN NumpyKNNRegressor with numpy, explicitly inherits from NumpyKNNBase already.

    Attributes:
        n_neighbors: A int number, number of neighbors.
        _metric: A method object, choose from {_manhattan_distance, _euclidean_distance, _chebyshev_distance}.
        _X_train: feature data for training. A np.ndarray matrix of (sample_lenght, feature_lenght) shape,
            data type must be continuous value type.
        _y_train: label data for training. A np.ndarray array of (sample_lenght, ) shape,
            data type must be discrete value.
    """

    def __new__(cls):
        return object.__new__(cls)

    def score(self, X_test, y_test):
        """Use test dataset to evaluate the trained model.

        Args:
            X_test: A np.ndarray matrix of (sample_lenght, feature_lenght) shape.
            y_test: A np.ndarray array of (sample_lenght, ) shape. data type must be
                discrete value.
        Returns:
            return R^2, R^2 = 1 - u / v. u = sum((y_pred - y_true)^2), v = sum((y_true - y_true_mean)^2)
        """
        X_test, y_test = self._score_validation(X_test, y_test)

        y_pred = self.predict(X_test)
        y_true_mean = np.mean(y_test, axis=0)
        u = np.sum(np.square(y_pred - y_test))
        v = np.sum(np.square(y_test - y_true_mean))
        r_squared = 1 - u / v
        return r_squared

    def _predict_sample(self, sample):
        k_labels = self._find_k_labels(sample)
        pred = np.mean(k_labels, axis=0)
        return pred


class TFKNNRegressor(TFKNNBase):
    """kNN Regressor with tensorflow, explicitly inherits from TFKNNBase already.

    Attributes:
        n_neighbors: A int number, number of neighbors.
        _metric: A method object, choose from {_manhattan_distance, _euclidean_distance, _chebyshev_distance}.
        _X_train: feature data for training. A tf.Tensor matrix of (sample_lenght, feature_lenght) shape,
            data type must be continuous value type.
        _y_train: label data for training. A tf.Tensor array of (sample_lenght, ) shape,
            data type must be discrete value.
    """

    def __new__(cls):
        return object.__new__(cls)

    def score(self, X_test, y_test):
        """Use test dataset to evaluate the trained model.

        Args:
            X_test: A np.ndarray matrix of (sample_lenght, feature_lenght) shape.
            y_test: A np.ndarray array of (sample_lenght, ) shape. data type must be
                discrete value.
        Returns:
            return R^2, R^2 = 1 - u / v. u = sum((y_pred - y_true)^2), v = sum((y_true - y_true_mean)^2)
        """
        X_test, y_test = self._score_validation(X_test, y_test)

        y_pred = self.predict(X_test)
        y_true_mean = tf.reduce_mean(y_test, axis=0)
        u = tf.reduce_sum(tf.square(y_pred - y_test))
        v = tf.reduce_sum(tf.square(y_test - y_true_mean))
        r_squared = 1 - u / v
        return r_squared.numpy()

    def _predict_sample(self, sample):
        k_labels = self._find_k_labels(sample)
        pred = tf.reduce_mean(k_labels, axis=0)
        return pred


class TorchKNNRegressor(TorchKNNBase):
    """kNN Regressor with Pytorch, explicitly inherits from TorchKNNBase already.

    Attributes:
        n_neighbors: A int number, number of neighbors.
        _metric: A method object, choose from {_manhattan_distance, _euclidean_distance, _chebyshev_distance}.
        _X_train: feature data for training. A torch.Tensr matrix of (sample_lenght, feature_lenght) shape,
            data type must be continuous value type.
        _y_train: label data for training. A torch.Tensor array of (sample_lenght, ) shape,
            data type must be discrete value.
    """

    def __new__(cls):
        return object.__new__(cls)

    def score(self, X_test, y_test):
        """Use test dataset to evaluate the trained model.

        Args:
            X_test: A np.ndarray matrix of (sample_lenght, feature_lenght) shape.
            y_test: A np.ndarray array of (sample_lenght, ) shape. data type must be
                discrete value.
        Returns:
            return R^2, R^2 = 1 - u / v. u = sum((y_pred - y_true)^2), v = sum((y_true - y_true_mean)^2)
        """
        X_test, y_test = self._score_validation(X_test, y_test)

        y_pred = torch.tensor(self.predict(X_test))
        y_true_mean = torch.mean(y_test)
        u = torch.sum(torch.square(y_pred - y_test), dim=0)
        v = torch.sum(torch.square(y_test - y_true_mean), dim=0)
        r_squared = 1 - u / v
        return r_squared.item()

    def _predict_sample(self, sample):
        k_labels = self._find_k_labels(sample)
        pred = torch.mean(k_labels)
        return pred.item()
