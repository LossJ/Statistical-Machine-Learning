from collections.abc import Iterable
from functools import reduce

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


class NumpyStandardScaler:
    """Standard Normalization Scaler with Numpy

        Attributes:
            mean_: The mean vector of features, with (feature_count, ) shape.
            scale_: The scale vector of features, with (feature_count, ) shape.
        """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X_train):
        """Fit the scaler from X_train data.

        Args:
            X_train: The train data of features, with (sample_count, feature_count) shape.
        """
        self.mean_ = np.mean(X_train, axis=0)
        self.scale_ = np.sqrt(np.mean(np.square(X_train - self.mean_), axis=0))

    def transform(self, X):
        """Transform X data.

        Args:
            X: The data of features, with (sample_count, feature_count) shape.

        Returns:
            A new matrix with transformation, with the same shape of old X.
        """
        return (X - self.mean_) / (self.scale_ + 0.000001)

    def fit_transform(self, X_train):
        """Fit and Transform X_train data.

        Args:
            X_train: The train data of features, with (sample_count, feature_count) shape.

        Returns:
            A new matrix with transformation, with the same shape of old X_train.
        """
        self.fit(X_train)
        return self.transform(X_train)


class NumpyPolynomialFeatures:
    """Polynomial Features transformor with Numpy.

    Attributes:
        degree: degree of power. A positive int number and
            must be greater than 1, default = 2.
        interaction_only: if only transform to the interaction.
            A bool value, default = False.
        include_bias: if include bias(intercept). A bool value,
            default = true.
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        """NumpyPolynomialFeatures initial method.

        Args:
            degree: degree of power. A positive int number and
                must be greater than 1, default = 2.
            interaction_only: if only transform to the interaction.
                A bool value, default = False.
            include_bias: if include bias(intercept). A bool value,
                default = true.
        """
        self.degree, self.interaction_only, self.include_bias = self._init_validation(degree, interaction_only,
                                                                                      include_bias)
        self._feature_func_dict = {}
        self._is_fitted = False

    def fit(self, X_train):
        """fit method.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape,
                data type must be continuous value type.
        Returns:
            return self object.
        """
        X_train = self._X_validation(X_train)
        if self._is_fitted:
            self.__init__(self.degree, self.interaction_only, self.include_bias)
        for i in range(X_train.shape[1]):
            self._feature_func_dict[(i,)] = lambda X: X[:, i].reshape([-1, 1])
        last_end = 0
        for loop in range(self.degree - 1):
            for j in range(last_end, len(self._feature_func_dict)):
                for k in range(0, X_train.shape[1]):
                    key_list = [*self._feature_func_dict]
                    if not (key_list[k][0] in key_list[j] and self.interaction_only):
                        new_key = self._sorted_insert_tuple(key_list[k][0], key_list[j])
                        if new_key not in self._feature_func_dict:
                            self._feature_func_dict[new_key] = lambda X: reduce(
                                lambda x, y: (X_train[:, x] if isinstance(x, int) else x) * X_train[:, y],
                                new_key).reshape([-1, 1])
                last_end += 1
        if self.include_bias:
            self._feature_func_dict["bias"] = lambda X: np.ones([X.shape[0], 1])
        self._is_fitted = True
        return self

    def transform(self, X, _miss_valid=False):
        """transform the data.

        Args:
            X: A np.ndarray matrix of (n_samples, n_features) shape,
                data type must be continuous value type.
        Returns:
            return a new X, a np.ndarray matrix of (n_samples,
                new_n_features) shape.
        """
        if not _miss_valid:
            X = self._X_validation(X)
        new_X = np.array([[] for _ in range(X.shape[0])])
        for key in self._feature_func_dict:
            new_X = np.c_[new_X, self._feature_func_dict[key](X)]
        return new_X

    def fit_transform(self, X_train):
        """fit transformer and transform data.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape,
                data type must be continuous value type.
        Returns:
            return a new X, a np.ndarray matrix of (n_samples,
                new_n_features) shape.
        """
        self.fit(X_train)
        return self.transform(X_train, _miss_valid=True)

    def get_feature_names(self):
        """get the new features' name

        Returns:
            return a list with the tuples of old feature's num.
                e.g: [(0,), (0, 0)], [(0,), (1,), (0, 0), (0, 1), (1, 1)].
        """
        return [*self._feature_func_dict]

    def _init_validation(self, degree, interaction_only, include_bias):
        assert isinstance(degree, int) and degree >= 1
        assert isinstance(interaction_only, bool)
        assert isinstance(include_bias, bool)
        return degree, interaction_only, include_bias

    def _X_validation(self, X):
        assert isinstance(X, Iterable)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        assert X.ndim == 2
        return X

    def _sorted_insert_tuple(self, item, tuple_):
        idx = len(tuple_)
        for i in range(len(tuple_)):
            if tuple_[i] < item:
                i += 1
            else:
                idx = i
                break
        tuple_ = tuple_[:idx] + (item,) + tuple_[idx:]
        return tuple_

