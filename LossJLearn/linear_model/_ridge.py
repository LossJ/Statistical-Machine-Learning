from ._base import NumpyBaseLinearRegressor, TFBaseLinearRegressor, TorchBaseLinearRegressor

import numpy as np
import tensorflow as tf
import torch


class NumpyRidge(NumpyBaseLinearRegressor):
    """Ridge class with numpy. explicitly inherits from NumpyBaseLinearRegressor already.

    Attributes:
        _X_train:feature data for training. A np.ndarray matrix of (n_samples, n_features) shape,
            data type must be continuous value type.
        _y_train:label data for training. A np.ndarray array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A np.ndarray array of (n_features, ) shape.
        intercept_: intercept of regressor. A np.ndarray integer.
        alpha: the regularize rate. A positive float number, default = 1.0.
    """

    def __init__(self, fit_intercept=True, alpha=1.0, random_state=None):
        """ridge object init method.

        Args:
            fit_intercept: Bool value. If use intercept in the ridge model.
            alpha: the regularize rate. A positive float number, default = 1.0.

        Raises:
            AssertionError: Alpha value must be a number.
            ValueError: Alpha value must be greater than 0.
        """
        super().__init__(fit_intercept=fit_intercept, random_state=random_state)
        assert isinstance(alpha, int) or isinstance(alpha, float)
        if 0 < alpha:
            self.alpha = alpha
        else:
            raise ValueError("Alpha value must be greater than 0! ")

    def _calculate_coef(self):
        xtx = self._X_train.T @ self._X_train
        self.coef_ = (
                np.linalg.inv(xtx + self.alpha * np.identity(self._X_train.shape[1])) \
                @ self._X_train.T \
                @ self._y_train
        )


class TFRidge(TFBaseLinearRegressor):
    """Ridge regressor class with tensorflow, explicitly inherits from TFBaseLinearRegressor already.

    Attributes:
        _X_train:feature data for training. A tf.Tensor matrix of (n_samples, n_features) shape,
            data type must be continuous value type.
        _y_train:label data for training. A tf.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A tf.Tensor matrix of (n_features, 1) shape.
        intercept_: intercept of regressor. A tf.Tensor integer if intercept_ is not None else None.
        alpha: the regularize rate. A positive float number, default = 1.0.
    """

    def __init__(self, fit_intercept=True, alpha=1.0, random_state=None):
        """ridge object init method.

        Args:
            fit_intercept: Bool value. If use intercept in the ridge model.
            alpha: the regularize rate. A positive float number, default = 1.0.

        Raises:
            AssertionError: Alpha value must be a number.
            ValueError: Alpha value must be greater than 0.
        """
        super().__init__(fit_intercept=fit_intercept, random_state=random_state)
        assert isinstance(alpha, int) or isinstance(alpha, float)
        if 0 < alpha:
            self.alpha = alpha
        else:
            raise ValueError("Alpha value must be greater than 0! ")

    def _calculate_coef(self):
        xtx = tf.transpose(self._X_train) @ self._X_train
        self.coef_ = (
                tf.linalg.inv(xtx + self.alpha * tf.eye(self._X_train.shape[1])) \
                @ tf.transpose(self._X_train) \
                @ tf.reshape(self._y_train, [-1, 1])
        )


class TorchRidge(TorchBaseLinearRegressor):
    """Ridge class with PyTorch. explicitly inherits from TorchBaseLinearRegressor already.

    Attributes:
        _X_train:feature data for training. A torch.Tensor matrix of (n_samples, n_features)
            shape, data type must be continuous value type.
        _y_train:label data for training. A torch.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A torch.Tensor array of (n_features, ) shape.
        intercept_: intercept of regressor. A torch.Tensor float number.
        alpha: the regularize rate. A positive float number, default = 1.0.
    """

    def __init__(self, fit_intercept=True, alpha=1.0, random_state=None):
        """ridge object init method.

        Args:
            fit_intercept: Bool value. If use intercept in the ridge model.
            alpha: the regularize rate. A positive float number, default = 1.0.

        Raises:
            AssertionError: Alpha value must be a number.
            ValueError: Alpha value must be greater than 0.
        """
        super().__init__(fit_intercept=fit_intercept, random_state=random_state)
        assert isinstance(alpha, int) or isinstance(alpha, float)
        if 0 < alpha:
            self.alpha = alpha
        else:
            raise ValueError("Alpha value must be greater than 0! ")

    def _calculate_coef(self):
        xtx = self._X_train.T @ self._X_train
        self.coef_ = (
                torch.inverse(xtx + self.alpha * torch.eye(self._X_train.shape[1])) \
                @ self._X_train.T \
                @ self._y_train
        )
