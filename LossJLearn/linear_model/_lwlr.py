from ._base import NumpyBaseLinearRegressor, TFBaseLinearRegressor, TorchBaseLinearRegressor

import numpy as np
import tensorflow as tf
import torch


class NumpyLWLR(NumpyBaseLinearRegressor):
    """Locally Weighted Linear Regression model with Numpy, explicitly inherits
        from NumpyBaseLinearRegressor already.

    Attributes:
        _X_train: feature data for training. A np.ndarray matrix of (n_samples, n_features)
            shape, data type must be continuous value type.
        _y_train: label data for training. A np.ndarray array of (n_samples, ) shape,
            data type must be continuous value.
        k: the k of gaussian kernal. A float number not 0, default = 1.0
    """

    def __init__(self, k=1.0, fit_intercept=True, random_state=None):
        """NumpyLWLR initial method

        Args:
            k: the k of gaussian kernal. A float number, default = 1.0
            fit_intercept: if fit intercept. A bool value, default = True.
            random_state: random seed. A int number if random_state is
                not None else None, default = None.
        """
        super().__init__(fit_intercept=fit_intercept, random_state=random_state)
        self.k = self._init_validation_self(k)

    def fit(self, X_train, y_train):
        """method for training model.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape, data type
                must be continuous value type.
            y_train: A np.ndarray array of (n_samples, ) shape.

        Returns:
            self object.
        """
        self._X_train, self._y_train = self._fit_validation(X_train, y_train)
        return self

    def predict(self, X_test, _miss_valid=False):
        """predict test data.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape,
                or a np.ndarray array of (n_features, ) shape.

        Returns:
            result of predict. A np.ndarray array of (n_samples, ) shape,
                or a float number.
        """
        if not _miss_valid:
            X_test = self._predict_validation(X_test)
        if self.intercept_ is not None:
            self._X_train = np.c_[self._X_train, np.ones(shape=(self._X_train.shape[0], 1), dtype=np.float32)]
            X_test = np.c_[X_test, np.ones((X_test.shape[0], 1))]
        if X_test.ndim == 1:
            result = self._predict_one(X_test)
            if self.intercept_ is not None:
                self._X_train = np.delete(self._X_train, -1, axis=1)
            return result
        else:
            results = []
            for x_test in X_test:
                results.append(self._predict_one(x_test))
            if self.intercept_ is not None:
                self._X_train = np.delete(self._X_train, -1, axis=1)
            return np.array(results)

    def _predict_one(self, x_test):
        w = self._calculate_weight(x_test)
        xtx = self._X_train.T @ (w @ self._X_train)
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError as ex:
            raise ex
        self.coef_ = xtx_inv @ self._X_train.T @ w @ self._y_train
        return x_test @ self.coef_

    def _init_validation_self(self, k):
        assert isinstance(k, (int, float))
        return k

    def _calculate_weight(self, x_test):
        w = np.eye(self._X_train.shape[0])
        for i in range(self._X_train.shape[0]):
            diff = self._X_train[i] - x_test
            w[i, i] = np.exp(diff @ diff.T / (-2 * self.k ** 2))
        return w


class TFLWLR(TFBaseLinearRegressor):
    """Locally Weighted Linear Regression model with TensorFlow, explicitly inherits
        from TFBaseLinearRegressor already.

    Attributes:
        _X_train: feature data for training. A tf.Tensor matrix of (n_samples, n_features)
            shape, data type must be continuous value type.
        _y_train: label data for training. A tf.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        k: the k of gaussian kernal. A float number not 0, default = 1.0
    """

    def __init__(self, k=1.0, fit_intercept=True, random_state=None):
        """TFLWLR initial method

        Args:
            k: the k of gaussian kernal. A float number, default = 1.0
            fit_intercept: if fit intercept. A bool value, default = True.
            random_state: random seed. A int number if random_state is
                not None else None, default = None.
        """
        super().__init__(fit_intercept=fit_intercept, random_state=random_state)
        self.k = self._init_validation_self(k)

    def fit(self, X_train, y_train):
        """method for training model.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape, data type
                must be continuous value type.
            y_train: A np.ndarray array of (n_samples, ) shape.

        Returns:
            self object.
        """
        self._X_train, self._y_train = self._fit_validation(X_train, y_train)
        return self

    def predict(self, X_test, _miss_valid=False):
        """predict test data.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape,
                or a np.ndarray array of (n_features, ) shape.

        Returns:
            result of predict. A tf.Tensor array of (n_samples, ) shape,
                or a float number.
        """
        if not _miss_valid:
            X_test = self._predict_validation(X_test)
        if self.intercept_ is not None:
            self._X_train = tf.concat([self._X_train, tf.ones(shape=(self._X_train.shape[0], 1))], axis=1)
            X_test = tf.concat([X_test, tf.ones(shape=(X_test.shape[0], 1))], axis=1)
        if X_test.ndim == 1:
            result = self._predict_one(X_test)
            if self.intercept_ is not None:
                self._X_train = self._X_train[:, :-1]
            return result
        else:
            results = []
            for x_test in X_test:
                results.append(self._predict_one(x_test).numpy())
            if self.intercept_ is not None:
                self._X_train = self._X_train[:, :-1]
            return tf.constant(results)

    def _predict_one(self, x_test):
        w = self._calculate_weight(x_test)
        xtx = tf.transpose(self._X_train) @ (w @ self._X_train)
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError as ex:
            raise ex
        self.coef_ = xtx_inv @ tf.transpose(self._X_train) @ w @ tf.reshape(self._y_train, shape=[-1, 1])
        return tf.reshape(tf.reshape(x_test, [1, -1]) @ self.coef_, [])

    def _init_validation_self(self, k):
        assert isinstance(k, (int, float))
        return k

    def _calculate_weight(self, x_test):
        w = tf.Variable(tf.eye(self._X_train.shape[0]))
        for i in range(self._X_train.shape[0]):
            diff = self._X_train[i] - x_test
            w[i, i].assign(tf.exp(tf.reduce_sum(tf.square(diff)) / (-2 * self.k * 2)))
        return w


class TorchLWLR(TorchBaseLinearRegressor):
    """Locally Weighted Linear Regression model with PyTorch, explicitly inherits
        from TorchBaseLinearRegressor already.

    Attributes:
        _X_train: feature data for training. A torch.Tensor matrix of (n_samples, n_features)
            shape, data type must be continuous value type.
        _y_train: label data for training. A torch.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        k: the k of gaussian kernal. A float number not 0, default = 1.0
    """

    def __init__(self, k=1.0, fit_intercept=True, random_state=None):
        """NumpyLWLR initial method

        Args:
            k: the k of gaussian kernal. A float number, default = 1.0
            fit_intercept: if fit intercept. A bool value, default = True.
            random_state: random seed. A int number if random_state is
                not None else None, default = None.
        """
        super().__init__(fit_intercept=fit_intercept, random_state=random_state)
        self.k = self._init_validation_self(k)

    def fit(self, X_train, y_train):
        """method for training model.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape, data type
                must be continuous value type.
            y_train: A np.ndarray array of (n_samples, ) shape.

        Returns:
            self object.
        """
        self._X_train, self._y_train = self._fit_validation(X_train, y_train)
        return self

    def predict(self, X_test, _miss_valid=False):
        """predict test data.

        Args:
            X_test: A np.ndarray matrix of (n_samples, n_features) shape,
                or a np.ndarray array of (n_features, ) shape.

        Returns:
            result of predict. A np.ndarray array of (n_samples, ) shape,
                or a float number.
        """
        if not _miss_valid:
            X_test = self._predict_validation(X_test)
        if self.intercept_ is not None:
            self._X_train = torch.cat((self._X_train, torch.ones(self._X_train.shape[0], 1, dtype=torch.float32)),
                                      dim=1)
            X_test = torch.cat((X_test, torch.ones(X_test.shape[0], 1, dtype=torch.float32)), dim=1)
        if X_test.ndim == 1:
            result = self._predict_one(X_test)
            if self.intercept_ is not None:
                self._X_train = self._X_train[:, :-1]
            return result
        else:
            results = []
            for x_test in X_test:
                results.append(self._predict_one(x_test).item())
            if self.intercept_ is not None:
                self._X_train = self._X_train[:, :-1]
            return torch.tensor(results)

    def _predict_one(self, x_test):
        w = self._calculate_weight(x_test)
        xtx = self._X_train.T @ (w @ self._X_train)
        try:
            xtx_inv = torch.inverse(xtx)
        except RuntimeError as ex:
            raise ex
        self.coef_ = xtx_inv @ self._X_train.T @ w @ self._y_train
        return x_test @ self.coef_

    def _init_validation_self(self, k):
        assert isinstance(k, (int, float))
        return k

    def _calculate_weight(self, x_test):
        w = torch.eye(self._X_train.shape[0])
        for i in range(self._X_train.shape[0]):
            diff = self._X_train[i] - x_test
            w[i, i] = torch.exp(diff @ diff.T / (-2 * self.k ** 2))
        return w
