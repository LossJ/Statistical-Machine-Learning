from collections.abc import Iterable

import numpy as np
import tensorflow as tf
import torch


class NumpyBaseEstimator:
    """Numpy estimator base class for all numpy estimator."""

    def _fit_validation(self, X, y):
        """valid fit method's params

        Args:
            X: feature data for training. A np.ndarray matrix of (n_samples,
                n_features) shape, data type must be continuous value type.
            y: label data for training. A np.ndarray array of (n_samples, ) shape.

        Returns:
            np.ndarray type data of X and y.

        Raises:
            AssertionError: Error type of X or y, or mismatched shape for X or y.
        """
        assert isinstance(X, Iterable) and isinstance(y, Iterable)
        assert len(X) == len(y)
        X = X if isinstance(X, np.ndarray) else np.array(X)
        y = y if isinstance(y, np.ndarray) else np.array(y)
        if X.ndim != 2:
            raise ValueError(f"X dim must be 2, got {X.ndim} dim of X! ")
        return X, y

    def _predict_validation(self, X_test):
        """valid predict method's params

        Args:
            X_test: feature data for predicting. A np.ndarray matrix of (n_samples,
                n_features) shape, data type must be continuous value type.

        Returns:
            np.ndarray type data of X_test.

        Raises:
            AssertionError: Error type of X_test.
            ValueError: Mismatched shape for X_test.
        """
        assert isinstance(X_test, Iterable)
        X_test = X_test if isinstance(X_test, np.ndarray) else np.array(X_test)
        error_text = f"Mismatched shape for X_test! Get {X_test.shape}, only" \
                     f" ({self._X_train.shape[1]}, ) or (_, {self._X_train.shape[1]}) " \
                     f"X_test can be."
        if X_test.ndim == 1:
            if X_test.shape[0] != self._X_train.shape[1]:
                raise ValueError(error_text)
        elif X_test.ndim == 2:
            if X_test.shape[1] != self._X_train.shape[1]:
                raise ValueError(error_text)
        else:
            raise ValueError(f"X_test with the {X_test.ndim} dim! Only 1 or 2 dim X_test can be.")
        return X_test

    def _score_validation(self, X_test, y_test):
        """valid score method's params

        Args:
            X_test: feature data for training. A np.ndarray matrix of (n_samples,
                n_features) shape, data type must be continuous value type.
            y_test: label data for training. A np.ndarray array of (n_samples, ) shape.

        Returns:
            np.ndarray type data of X_test and y_test.

        Raises:
            AssertionError: Error type of X_test or y_test.
            ValueError: Mismatched shape for X_test or y_test.
        """
        X_test = self._predict_validation(X_test)
        is_valid = True
        if isinstance(y_test, Iterable):
            if X_test.shape[0] != len(y_test):
                is_valid = False
            elif X_test.ndim == 1:
                is_valid = False
        else:
            assert isinstance(y_test, int) or isinstance(y_test, float)
            if X_test.ndim != 1:
                is_valid = False
        y_test = np.array(y_test)
        if not is_valid:
            raise ValueError(f"Mismatched shape for y_test! X_test with {X_test.shape} shape, but "
                             f"y_test's shape is {y_test.shape}")
        return X_test, y_test


class TFBaseEstimator(NumpyBaseEstimator):
    """TensorFlow estimator base class for all TensorFlow estimator."""
    def _fit_validation(self, X, y):
        """valid fit method's params

        Args:
            X: feature data for training. A np.ndarray matrix of (n_samples,
                n_features) shape, data type must be continuous value type.
            y: label data for training. A np.ndarray array of (n_samples, ) shape.

        Returns:
            tf.Tensor type data of X and y.

        Raises:
            AssertionError: Error type of X or y, or mismatched shape for X or y.
        """
        X, y = super()._fit_validation(X, y)
        return tf.constant(X, dtype=tf.float32), tf.constant(y, dtype=tf.float32)

    def _predict_validation(self, X_test):
        """valid predict method's params

        Args:
            X_test: feature data for predicting. A np.ndarray matrix of (n_samples,
                n_features) shape, data type must be continuous value type.

        Returns:
            tf.Tensor type data of X_test.

        Raises:
            AssertionError: Error type of X_test.
            ValueError: Mismatched shape for X_test.
        """
        X_test  = super()._predict_validation(X_test)
        return tf.constant(X_test, dtype=tf.float32)

    def _score_validation(self, X_test, y_test):
        """valid score method's params

        Args:
            X_test: feature data for training. A np.ndarray matrix of (n_samples,
                n_features) shape, data type must be continuous value type.
            y_test: label data for training. A np.ndarray array of (n_samples, ) shape.

        Returns:
            tf.Tensor type data of X_test and y_test.

        Raises:
            AssertionError: Error type of X_test or y_test.
            ValueError: Mismatched shape for X_test or y_test.
        """
        X_test, y_test = super()._score_validation(X_test, y_test)
        return tf.constant(X_test, dtype=tf.float32), tf.constant(y_test, dtype=tf.float32)


class TorchBaseEstimator(NumpyBaseEstimator):
    """PyTorch estimator base class for all PyTorch estimator."""

    def _fit_validation(self, X, y):
        """valid fit method's params

        Args:
            X: feature data for training. A np.ndarray matrix of (n_samples,
                n_features) shape, data type must be continuous value type.
            y: label data for training. A np.ndarray array of (n_samples, ) shape.

        Returns:
            torch.Tensor type data of X and y.

        Raises:
            AssertionError: Error type of X or y, or mismatched shape for X or y.
        """
        X, y = super()._fit_validation(X, y)
        return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)

    def _predict_validation(self, X_test):
        """valid predict method's params

        Args:
            X_test: feature data for predicting. A np.ndarray matrix of (n_samples,
                n_features) shape, data type must be continuous value type.

        Returns:
            torch.Tensor type data of X_test.

        Raises:
            AssertionError: Error type of X_test.
            ValueError: Mismatched shape for X_test.
        """
        X_test = super()._predict_validation(X_test)
        return torch.as_tensor(X_test, dtype=torch.float32)

    def _score_validation(self, X_test, y_test):
        """valid score method's params

        Args:
            X_test: feature data for training. A np.ndarray matrix of (n_samples,
                n_features) shape, data type must be continuous value type.
            y_test: label data for training. A np.ndarray array of (n_samples, ) shape.

        Returns:
            torch.Tensor type data of X_test and y_test.

        Raises:
            AssertionError: Error type of X_test or y_test.
            ValueError: Mismatched shape for X_test or y_test.
        """
        X_test, y_test = super()._score_validation(X_test, y_test)
        return torch.as_tensor(X_test, dtype=torch.float32), torch.as_tensor(y_test, dtype=torch.float32)


class NumpyRegressorMixin:
    """Regressor Mixin with numpy. Mainly realized the score method."""

    def score(self, X_test, y_test):
        """Use test dataset to evaluate the trained model.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape.
            y_test: A np.ndarray array of (n_samples, ) shape. data type must be
                discrete value.
        Returns:
            return R^2, R^2 = 1 - u / v. u = sum((y_pred - y_true)^2), v = sum((y_true - y_true_mean)^2)
        """
        X_test, y_test = self._score_validation(X_test, y_test)
        y_pred = self.predict(X_test, _miss_valid=True)
        y_true_mean = np.mean(y_test, axis=0)
        u = np.sum(np.square(y_pred - y_test))
        v = np.sum(np.square(y_test - y_true_mean))
        r_squared = 1 - u / v
        return r_squared


class TFRegressorMixin:
    """Regressor Mixin with TensorFlow. Mainly realized the score method."""

    def score(self, X_test, y_test):
        """Use test dataset to evaluate the trained model.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape.
            y_test: A np.ndarray array of (n_samples, ) shape. data type must be
                discrete value.
        Returns:
            return R^2, a np.ndarray float number. R^2 = 1 - u / v.
                u = sum((y_pred - y_true)^2), v = sum((y_true - y_true_mean)^2)
        """
        X_test, y_test = self._score_validation(X_test, y_test)
        y_pred = self.predict(X_test, _miss_valid=True)
        y_true_mean = tf.reduce_mean(y_test, axis=0)
        u = tf.reduce_sum(tf.square(y_pred - y_test))
        v = tf.reduce_sum(tf.square(y_test - y_true_mean))
        r_squared = 1 - u / v
        return r_squared.numpy()

class TorchRegressorMixin:
    """Regressor Mixin with PyTorch. Mainly realized the score method."""

    def score(self, X_test, y_test):
        """Use test dataset to evaluate the trained model.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape.
            y_test: A np.ndarray array of (n_samples, ) shape. data type must be
                discrete value.
        Returns:
            return R^2, a np.ndarray float number. R^2 = 1 - u / v.
                u = sum((y_pred - y_true)^2), v = sum((y_true - y_true_mean)^2)
        """
        X_test, y_test = self._score_validation(X_test, y_test)
        y_pred = self.predict(X_test, _miss_valid=True)
        y_true_mean = torch.mean(y_test, axis=0)
        u = torch.sum(torch.square(y_pred - y_test))
        v = torch.sum(torch.square(y_test - y_true_mean))
        r_squared = 1 - u / v
        return r_squared.item()
