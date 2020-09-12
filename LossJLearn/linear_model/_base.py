from ..base import NumpyBaseEstimator, NumpyRegressorMixin, TFBaseEstimator, TFRegressorMixin, TorchBaseEstimator, TorchRegressorMixin

import numpy as np
import tensorflow as tf
import torch

class NumpyBaseLinearRegressor(NumpyBaseEstimator, NumpyRegressorMixin):
    """Linear regressor base class with numpy, explicitly inherits from NumpyBaseEstimator
        and NumpyRegressorMixin already.

    Attributes:
        _X_train:feature data for training. A np.ndarray matrix of (n_samples,
            n_features) shape, data type must be continuous value type.
        _y_train:label data for training. A np.ndarray array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A np.ndarray array of (n_features, ) shape.
        intercept_: intercept of regressor. A np.ndarray integer.
    """

    def __init__(self, fit_intercept=True, random_state=None):
        """linear regressor init method.

        Args:
            fit_intercept: Bool value. If use intercept in the linear regressor model.
        """
        fit_intercept, random_state = self._init_validation(fit_intercept, random_state)
        self._X_train = None
        self._y_train = None
        self.coef_ = None
        if random_state is not None:
            np.random.seed(random_state)
        self.intercept_ = np.random.randn() if fit_intercept else None

    def fit(self, X_train, y_train):
        """method for training model.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape,
                data type must be continuous value type.
            y_train: A np.ndarray array of (n_samples, ) shape.

        Returns:
            self object.
        """
        self._X_train, self._y_train = self._fit_validation(X_train, y_train)
        if self.intercept_ is not None:
            self._X_train = np.c_[self._X_train, np.ones((self._X_train.shape[0], 1))]
        self._calculate_coef()
        if self.intercept_ is not None:
            self.intercept_ = self.coef_[-1]
            self.coef_ = np.delete(self.coef_, -1)
            self._X_train = np.delete(self._X_train, -1, axis=1)
        return self

    def predict(self, X_test, _miss_valid=False):
        """predict test data.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape,
                or a np.ndarray array of (n_features, ) shape.
        """
        if not _miss_valid:
            X_test = self._predict_validation(X_test)
        product = X_test @ self.coef_
        return product if self.intercept_ is None else product + self.intercept_

    def _init_validation(self, fit_intercept, random_state):
        assert isinstance(fit_intercept, bool)
        assert isinstance(random_state, (int, type(None)))
        return fit_intercept, random_state


class TFBaseLinearRegressor(TFBaseEstimator, TFRegressorMixin):
    """Linear regressor base class with tensorflow, explicitly inherits
        from TFBaseEstimator and TFRegressorMixin already.

    Attributes:
        _X_train:feature data for training. A tf.Tensor matrix of (n_samples, n_features) shape,
            data type must be continuous value type.
        _y_train:label data for training. A tf.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A tf.Tensor matrix of (n_features, 1) shape.
        intercept_: intercept of regressor. A tf.Tensor integer if intercept_ is not None else None.
    """

    def __init__(self, fit_intercept=True, random_state=None):
        """linear regressor init method.

        Args:
            fit_intercept: Bool value. If use intercept in the linear regressor model.
        """
        self._X_train = None
        self._y_train = None
        self.coef_ = None
        if random_state:
            assert isinstance(random_state, int)
            tf.random.set_seed = random_state
        self.intercept_ = tf.random.normal(shape=[]) if fit_intercept else None

    def fit(self, X_train, y_train):
        """method for training model.

        Args:
            X_train: A tf.Tensor matrix of (n_samples, n_features) shape, data type must be continuous value type.
            y_train: A tf.Tensor array of (n_samples, ) shape.

        Returns:
            self object.
        """
        self._X_train, self._y_train = self._fit_validation(X_train, y_train)
        if self.intercept_:
            self._X_train = tf.concat(
                [self._X_train, tf.ones([self._X_train.shape[0], 1])], axis=1
            )
        self._calculate_coef()
        if self.intercept_:
            self.intercept_ = self.coef_[-1][0]
            self._X_train = self._X_train[:, :-1]
            self.coef_ = self.coef_[:-1]
        return self

    def predict(self, X_test, _miss_valid=False):
        """predict test data.

        Args:
            X_test: A tf.Tensor matrix of (n_samples, n_features) shape,
                or a tf.Tensor array of (n_features, ) shape.

        Returns:
            result of predict. A tf.Tensor array of (n_samples, ) shape.
        """
        if not _miss_valid:
            X_test = self._predict_validation(X_test)
        product = X_test @ self.coef_
        result = product if self.intercept_ is None else product + self.intercept_
        return tf.reshape(result, [-1])


class TorchBaseLinearRegressor(TorchBaseEstimator, TorchRegressorMixin):
    """Linear regressor base class with PyTorch, explicitly inherits from
        TorchBaseEstimator and TorchRegressorMixin already.

    Attributes:
        _X_train:feature data for training. A torch.Tensor matrix of (n_samples,
            n_features) shape, data type must be continuous value type.
        _y_train:label data for training. A torch.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A torch.Tensor array of (n_features, ) shape.
        intercept_: intercept of regressor. A torch.Tensor float number.
    """

    def __init__(self, fit_intercept=True, random_state=None):
        """linear regressor init method.

        Args:
            fit_intercept: If use intercept in the linear regressor model.
                A bool value, default = True
        """
        fit_intercept, random_state = self._init_validation(fit_intercept, random_state)
        self._X_train = None
        self._y_train = None
        self.coef_ = None
        if random_state is not None:
            torch.manual_seed(random_state)
        self.intercept_ = torch.randn([]) if fit_intercept else None

    def fit(self, X_train, y_train):
        """method for training model.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape,
                data type must be continuous value type.
            y_train: A np.ndarray array of (n_samples, ) shape.

        Returns:
            self object.
        """
        self._X_train, self._y_train = self._fit_validation(X_train, y_train)
        if self.intercept_ is not None:
            self._X_train = torch.cat((self._X_train, torch.ones(self._X_train.shape[0], 1)), dim=1)
        self._calculate_coef()
        if self.intercept_ is not None:
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
            self._X_train = self._X_train[:, :-1]
        return self

    def predict(self, X_test, _miss_valid=False):
        """predict test data.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape,
                or a np.ndarray array of (n_features, ) shape.
        """
        if not _miss_valid:
            X_test = self._predict_validation(X_test)
        product = X_test @ self.coef_
        return product if self.intercept_ is None else product + self.intercept_

    def _init_validation(self, fit_intercept, random_state):
        assert isinstance(fit_intercept, bool)
        assert isinstance(random_state, (int, type(None)))
        return fit_intercept, random_state


class NumpyLinearRegressor(NumpyBaseLinearRegressor):
    """Linear regressor class with numpy, explicitly inherits from NumpyBaseLinearRegressor already.

    Attributes:
        _X_train:feature data for training. A np.ndarray matrix of (n_samples, n_features) shape,
            data type must be continuous value type.
        _y_train:label data for training. A np.ndarray array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A np.ndarray array of (n_features, ) shape.
        intercept_: intercept of regressor. A np.ndarray integer.
    """

    def _calculate_coef(self):
        xtx = self._X_train.T @ self._X_train
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError as ex:
            raise ex
        self.coef_ = xtx_inv @ self._X_train.T @ self._y_train


class TFLinearRegressor(TFBaseLinearRegressor):
    """Linear regressor class with tensorflow, explicitly inherits from TFBaseLinearRegressor already.

    Attributes:
        _X_train:feature data for training. A tf.Tensor matrix of (n_samples, n_features) shape,
            data type must be continuous value type.
        _y_train:label data for training. A tf.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A tf.Tensor matrix of (n_features, 1) shape.
        intercept_: intercept of regressor. A tf.Tensor integer if intercept_ is not None else None.
    """

    def _calculate_coef(self):
        xtx = tf.transpose(self._X_train) @ self._X_train
        try:
            xtx_inv = tf.linalg.inv(xtx)
        except tf.errors.InvalidArgumentError as ex:
            raise ex
        self.coef_ = (
                tf.linalg.inv(xtx) \
                @ tf.transpose(self._X_train) \
                @ tf.reshape(self._y_train, [-1, 1])
        )


class TorchLinearRegressor(TorchBaseLinearRegressor):
    """Linear regressor class with PyTorch, explicitly inherits from
        TorchBaseLinearRegressor already.

    Attributes:
        _X_train:feature data for training. A torch.Tensor matrix of (n_samples, n_features) shape,
            data type must be continuous value type.
        _y_train:label data for training. A torch.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A torch.Tensor array of (n_features, ) shape.
        intercept_: intercept of regressor. A torch.Tensor float number.
    """

    def _calculate_coef(self):
        xtx = self._X_train.T @ self._X_train
        try:
            xtx_inv = torch.inverse(xtx)
        except RuntimeError as ex:
            raise ex
        self.coef_ = xtx_inv @ self._X_train.T @ self._y_train
