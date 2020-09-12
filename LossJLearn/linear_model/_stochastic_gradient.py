import time
import copy

from ._base import NumpyBaseLinearRegressor, TFBaseLinearRegressor, TorchBaseLinearRegressor
from ..utils.translator import sec2time
from ..datasets._generator import TorchBaseDataset

import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


class NumpySGDBaseEstimator:
    """SGD Estimator Base class with numpy.

    Attributes:
        _X_train:feature data for training. A np.ndarray matrix of (n_samples,
            n_features) shape, data type must be continuous value type.
        _y_train:label data for training. A np.ndarray array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A np.ndarray matrix of (n_features, ) shape.
        intercept_: intercept of regressor. A np.ndarray integer if intercept_ is
            not None else None.
        alpha: the regularize rate. A float number and must be greater than 0,
            default = 0.001.
        save_param_list: if save param of the train process. A bool value, default = True.
        coef_list: list of coef param from the train process,
            every coef is a np.ndarray of (n_features, ) shape.
        intercept_list: list of intercept param from the train process,
            every intercept is a np.ndarray float number.
        learning_rate: learning rate. A positive float number, default = 0.001.
        epochs: epochs. A positive int number, default = 10.
        batch_size: batch size. A positive int number, default = 32.
        early_stopping: if early stopping when loss don't reduce again. A bool value,
            default = True.
        patient: Number of epochs that do not reduce loss continuously,
            patient only takes effect when early_stopping is True.
            A positive int number, default = 5.
        toc: The threshold that symbolizes loss no longer decreases,
            toc only takes effect when early_stopping is True.
            A float number, default = 0.001
        random_state: random seed. A positive int number if random_state
            is not None else None, default = None.
        regularize: regularize. A str value in {"l1", "l2"} if regularize
            is not None else None, default = None.
        best_loss: best loss of the train process. A np.ndarray float number.
        best_coef: best coef of the train process. A np.ndarray array of
            (n_features, 1) shape.
        best_intercept_: best intercept of the train process. A np.ndarray number.
        train_loss: list of train loss from the train process.
            every loss is a np.ndarray float number.
        valid_loss: list of valid loss from the train process.
            every loss is a np.ndarray float number.
        n_iter: the actual iteration of train process. A int number, initial = 0.
        save_best_model: if save the best model params as the final model.
                A bool value, defalut = True.
    """

    def __init__(
            self,
            loss="mse",
            alpha=0.0001,
            fit_intercept=True,
            save_param_list=True,
            learning_rate=0.0001,
            epochs=10,
            batch_size=32,
            print_step=1,
            early_stopping=True,
            patient=5,
            toc=0.0001,
            random_state=None,
            regularize=None,
            shuffle=True,
            save_best_model=True
    ):
        """NumpySGDBaseEstimator initial method.

        Args:
            loss: A str in {"mse"}, default = "mse"
            alpha: the regularize rate. A float number and must be greater
                than 0, default = 0.001.
            fit_intercept: if fit intercept. A bool value, default = True.
            save_param_list: if save param of the train process. A bool value,
                default = True.
            learning_rate: learning rate. A positive float number, default = 0.001.
            epochs: epochs. A positive int number, default = 10.
            batch_size: batch size. A positive int number, default = 32.
            early_stopping: if early stopping when loss don't reduce again.
                A bool value, default = True
            patient: Number of epochs that do not reduce loss continuously,
                patient only takes effect when early_stopping is True.
                A positive int number, default = 5.
            toc: The threshold that symbolizes loss no longer decreases,
                toc only takes effect when early_stopping is True.
                A float number, default = 0.001
            random_state: random seed. A positive int number if random_state
                is not None else None, default = None.
            regularize: regularize. A str value in {"l2"} if regularize
                is not None else None, default = None.
            shuffle: if shuffle the train data. A bool value, default = True.
            save_best_model: if save the best model params as the final model.
                A bool value, defalut = True.

        Raises:
            AssertionError: Some parameters do not match.
        """

        (
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            print_step,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            shuffle,
            save_best_model
        ) = self._init_validation(
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            print_step,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            shuffle,
            save_best_model
        )
        self.random_state = None
        if random_state:
            self.random_state = random_state
            np.random.seed(self.random_state)
        loss_func_dict = {"mse": self._mse}
        loss_gradient_func_dict = {"mse": self._mse_gradient}
        self._loss_func = loss_func_dict[loss]
        self._gradient_func = loss_gradient_func_dict[loss]
        self.alpha = alpha
        self.intercept_ = None
        if fit_intercept:
            self.intercept_ = np.random.randn()
        self.save_param_list = save_param_list
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self._print_step = print_step
        self.early_stopping = early_stopping
        self.patient = patient
        self.toc = toc
        self.regularize = regularize
        self.shuffle = shuffle
        self.save_best_model = save_best_model

        self._X_train = None
        self._y_train = None
        self.n_iter_ = 0
        self.coef_ = None
        self.best_loss = float("inf")
        self.best_coef_ = None
        self.best_intercept_ = None
        self.best_epoch = 1
        self.coef_list = []
        self.intercept_list = []
        self.valid_loss = []
        self.train_loss = []

    def fit(self, X_train, y_train, validation_data=None):
        """train model methed.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape,
                data type must be continuous value type.
            y_train: A np.ndarray array of (n_samples, ) shape, data type
                must be continuous value type.
            validation: the validation data for validate the model. A tuple
                like (X_valid, y_valid) , the shape of X_valid and y_valid is
                like X_train and y_train. Default = None.
        """
        self._X_train, self._y_train = self._fit_validation(X_train, y_train)
        X_valid, y_valid = self._validation_data_valid(validation_data)
        if self.coef_ is None:
            self.coef_ = np.random.randn(self._X_train.shape[1])
        current_patient = 0
        last_valid_loss = None
        self.coef_list.append(copy.deepcopy(self.coef_))
        self.intercept_list.append(self.intercept_)
        for epoch in range(self.epochs):
            # train model
            self._fit_train()

            # validate model
            valid_mean_loss = self._fit_valid(X_valid, y_valid, epoch)

            self.n_iter_ += 1

            # early stopping
            if self.early_stopping and epoch != 0:
                if last_valid_loss - valid_mean_loss < self.toc:
                    current_patient += 1
                else:
                    current_patient = 0
                if current_patient >= self.patient:
                    break
            last_valid_loss = valid_mean_loss
        if self.save_best_model:
            self.coef_ = self.best_coef_
            self.intercept_ = self.best_intercept_
        self._final_print()
        return self

    def _final_print(self):
        print(
            f"Actual iter epoch: {self.n_iter_}, best epoch: {self.best_epoch}, "
            f"best loss: {self.best_loss}, best coef: {self.best_coef_}, "
            f"best intercept: {self.best_intercept_}"
        )

    def _fit_train(self):
        train_data = self._batch_generator(self._X_train, self._y_train, self.shuffle)
        for X_batch, y_batch in train_data:
            y_pred = self.predict(X_batch, _miss_valid=True)
            coef_gradient, intercept_gradient = self._gradient_func(
                y_batch, y_pred, X_batch
            )
            self.coef_ -= self.learning_rate * coef_gradient
            if self.intercept_:
                self.intercept_ -= self.learning_rate * intercept_gradient
        train_loss_last_batch = self._loss_func(y_batch, y_pred)
        self.train_loss.append(train_loss_last_batch)

    def _fit_valid(self, X_valid, y_valid, epoch):
        valid_data = self._batch_generator(X_valid, y_valid, self.shuffle)
        valid_sum_loss = 0
        for X_valid_batch, y_valid_batch in valid_data:
            valid_batch_pred = self.predict(X_valid_batch)
            loss = self._loss_func(y_valid_batch, valid_batch_pred)
            valid_sum_loss += loss
        valid_mean_loss = valid_sum_loss / (X_valid.shape[0] // self.batch_size)
        if valid_mean_loss < self.best_loss:
            self.best_loss = valid_mean_loss
            self.best_coef_ = self.coef_
            self.best_intercept_ = self.intercept_
            self.best_epoch = epoch + 1
        if self.save_param_list:
            self.coef_list.append(copy.deepcopy(self.coef_))
            self.intercept_list.append(self.intercept_)
        if (epoch + 1) % self._print_step == 0:
            print(f"Epoch {epoch + 1}: valid_data loss: {valid_mean_loss}")
        self.valid_loss.append(valid_mean_loss)
        return valid_mean_loss

    def _validation_data_valid(self, validation_data):
        if validation_data:
            assert isinstance(validation_data, tuple) and len(validation_data) == 2
            X_valid, y_valid = validation_data
            X_valid, y_valid = self._fit_validation(X_valid, y_valid)
        else:
            X_train_len = int(self._X_train.shape[0] * 0.25)
            X_valid = self._X_train[X_train_len:]
            y_valid = self._y_train[X_train_len:]
            self._X_train = self._X_train[:X_train_len]
            self._y_train = self._y_train[:X_train_len]
        return X_valid, y_valid

    def _batch_generator(self, X_data, y_data, shuffle=True):
        step = 0
        steps_per_epoch = X_data.shape[0] // self.batch_size
        while steps_per_epoch > step:
            if shuffle:
                index = np.random.choice(X_data.shape[0], self.batch_size)
            else:
                index = np.arange(self.batch_size * step, self.batch_size * (step + 1))
            yield X_data[index], y_data[index]
            step += 1

    def _regularize_gradient(self, coef_gradient):
        if self.regularize is None:
            return coef_gradient
        elif self.regularize == "l2":
            return coef_gradient + 2 * self.alpha * self.coef_

    def _init_validation(
            self,
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            print_step,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            shuffle,
            save_best_model
    ):
        loss_key_set = {"mse", "cross_entropy"}
        assert loss in loss_key_set
        assert isinstance(alpha, (int, float))
        assert 0 < alpha
        assert isinstance(fit_intercept, bool)
        assert isinstance(save_param_list, bool)
        assert isinstance(learning_rate, (int, float))
        assert 0 < learning_rate
        assert isinstance(epochs, int) and epochs >= 1
        assert isinstance(batch_size, int) and batch_size >= 1
        assert isinstance(print_step, int) and print_step >= 1
        assert isinstance(early_stopping, bool)
        assert isinstance(patient, int) and patient >= 2
        assert isinstance(toc, (int, float)) and toc > 0.0
        assert (random_state is None) or isinstance(random_state, int)
        regularize_key_set = {None, "l2"}
        assert regularize in regularize_key_set
        assert isinstance(shuffle, bool)
        assert isinstance(save_best_model, bool)
        return (
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            print_step,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            shuffle,
            save_best_model
        )


class NumpySGDRegressor(NumpySGDBaseEstimator, NumpyBaseLinearRegressor):
    """SGD Regressor model with numpy, explicitly inherits
        from NumpyBaseLinearRegression and NumpySGDBaseEstimator already.

    Attributes:
        _X_train:feature data for training. A np.ndarray matrix of (n_samples,
            n_features) shape, data type must be continuous value type.
        _y_train:label data for training. A np.ndarray array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A np.ndarray matrix of (n_features, ) shape.
        intercept_: intercept of regressor. A np.ndarray integer if intercept_ is
            not None else None.
        alpha: the regularize rate. A float number and must be greater than 0,
            default = 0.001.
        save_param_list: if save param of the train process. A bool value, default = True.
        coef_list: list of coef param from the train process,
            every coef is a np.ndarray of (n_features, ) shape.
        intercept_list: list of intercept param from the train process,
            every intercept is a np.ndarray float number.
        learning_rate: learning rate. A positive float number, default = 0.001.
        epochs: epochs. A positive int number, default = 10.
        batch_size: batch size. A positive int number, default = 32.
        early_stopping: if early stopping when loss don't reduce again. A bool value,
            default = True.
        patient: Number of epochs that do not reduce loss continuously,
            patient only takes effect when early_stopping is True.
            A positive int number, default = 5.
        toc: The threshold that symbolizes loss no longer decreases,
            toc only takes effect when early_stopping is True.
            A float number, default = 0.001
        random_state: random seed. A positive int number if random_state
            is not None else None, default = None.
        regularize: regularize. A str value in {"l1", "l2"} if regularize
            is not None else None, default = None.
        best_loss: best loss of the train process. A np.ndarray float number.
        best_coef: best coef of the train process. A np.ndarray array of
            (n_features, 1) shape.
        best_intercept_: best intercept of the train process. A np.ndarray number.
        train_loss: list of train loss from the train process.
            every loss is a np.ndarray float number.
        valid_loss: list of valid loss from the train process.
            every loss is a np.ndarray float number.
        n_iter: the actual iteration of train process. A int number, initial = 0.
    """

    def __init__(
            self,
            loss="mse",
            alpha=0.0001,
            fit_intercept=True,
            save_param_list=True,
            learning_rate=0.0001,
            epochs=10,
            batch_size=32,
            print_step=1,
            early_stopping=True,
            patient=5,
            toc=0.0001,
            random_state=None,
            regularize=None,
            shuffle=True
    ):
        """NumpySGDRegressor initial method.

        Args:
            loss: A str in {"mse"}, default = "mse"
            alpha: the regularize rate. A float number and must be greater
                than 0, default = 0.001.
            fit_intercept: if fit intercept. A bool value, default = True.
            save_param_list: if save param of the train process. A bool value,
                default = True.
            learning_rate: learning rate. A positive float number, default = 0.001.
            epochs: epochs. A positive int number, default = 10.
            batch_size: batch size. A positive int number, default = 32.
            early_stopping: if early stopping when loss don't reduce again.
                A bool value, default = True
            patient: Number of epochs that do not reduce loss continuously,
                patient only takes effect when early_stopping is True.
                A positive int number, default = 5.
            toc: The threshold that symbolizes loss no longer decreases,
                toc only takes effect when early_stopping is True.
                A float number, default = 0.001
            random_state: random seed. A positive int number if random_state
                is not None else None, default = None.
            regularize: regularize. A str value in {"l2"} if regularize
                is not None else None, default = None.
            shuffle: if shuffle the train data. A bool value, default = True.
            save_best_model: if save the best model params as the final model.
                A bool value, defalut = True

        Raises:
            AssertionError: Some parameters do not match.
        """
        super().__init__(
            loss=loss,
            alpha=alpha,
            fit_intercept=fit_intercept,
            save_param_list=save_param_list,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            print_step=print_step,
            early_stopping=early_stopping,
            patient=patient,
            toc=toc,
            random_state=random_state,
            regularize=regularize,
            shuffle=shuffle
        )

    def _mse_gradient(self, y_true, y_pred, x):
        difference_y = y_pred - y_true
        intercept_gradient = None
        if self.intercept_:
            intercept_gradient = np.sum(difference_y) * 2 / self.batch_size
        coef_gradient = (
                np.sum(difference_y.reshape([-1, 1]) * x, axis=0) * 2 / self.batch_size
        )
        coef_gradient = self._regularize_gradient(coef_gradient)
        return coef_gradient, intercept_gradient

    def _mse(self, y, pred):
        return np.sum(np.square(y - pred)) / y.shape[0]


class TFSGDRegressor(TFBaseLinearRegressor):
    """Linear SGD regressor class with tensorflow, explicitly inherits
        from TFBaseLinearRegressor already.

    Attributes:
        _X_train:feature data for training. A tf.Tensor matrix of (n_samples,
            n_features) shape, data type must be continuous value type.
        _y_train:label data for training. A tf.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A tf.Tensor matrix of (n_features, 1) shape.
        intercept_: intercept of regressor. A tf.Tensor integer if intercept_ is
            not None else None.
        alpha: the regularize rate. A float number and must be greater than 0,
            default = 0.001.
        save_param_list: if save param of the train process. A bool value, default = True.
        coef_list: list of coef param from the train process,
            every coef is a np.ndarray of (n_features, ) shape.
        intercept_list: list of intercept param from the train process,
            every intercept is a np.ndarray float number.
        learning_rate: learning rate. A positive float number, default = 0.001.
        epochs: epochs. A positive int number, default = 10.
        batch_size: batch size. A positive int number, default = 32.
        early_stopping: if early stopping when loss don't reduce again. A bool value,
            default = True.
        patient: Number of epochs that do not reduce loss continuously,
            patient only takes effect when early_stopping is True.
            A positive int number, default = 5.
        toc: The threshold that symbolizes loss no longer decreases,
            toc only takes effect when early_stopping is True.
            A float number, default = 0.001
        random_state: random seed. A positive int number if random_state
            is not None else None, default = None.
        regularize: regularize. A str value in {"l1", "l2"} if regularize
            is not None else None, default = None.
        best_loss: best loss of the train process. A np.ndarray float number.
        best_coef: best coef of the train process. A tf.Tensor array of
            (n_features, 1) shape.
        best_intercept_: best intercept of the train process. A tf.Tensor number.
        train_loss: list of train loss from the train process.
            every loss is a np.ndarray float number.
        valid_loss: list of valid loss from the train process.
            every loss is a np.ndarray float number.
        n_iter: the actual iteration of train process. A int number, initial = 0.
        save_best_model: if save the best model params as the final model.
                A bool value, defalut = True.
    """

    def __init__(
            self,
            loss="mse",
            alpha=0.001,
            fit_intercept=True,
            save_param_list=True,
            learning_rate=0.001,
            epochs=10,
            batch_size=32,
            early_stopping=True,
            patient=5,
            toc=0.001,
            random_state=None,
            regularize=None,
            save_best_model=True
    ):
        """TFSGDRegressor initial method.

        Args:
            loss: A str in {"mse"}, default = "mse"
            alpha: the regularize rate. A float number and must be greater
                than 0, default = 0.001.
            fit_intercept: if fit intercept. A bool value, default = True.
            save_param_list: if save param of the train process. A bool value,
                default = True.
            learning_rate: learning rate. A positive float number, default = 0.001.
            epochs: epochs. A positive int number, default = 10.
            batch_size: batch size. A positive int number, default = 32.
            early_stopping: if early stopping when loss don't reduce again.
                A bool value, default = True
            patient: Number of epochs that do not reduce loss continuously,
                patient only takes effect when early_stopping is True.
                A positive int number, default = 5.
            toc: The threshold that symbolizes loss no longer decreases,
                toc only takes effect when early_stopping is True.
                A float number, default = 0.001
            random_state: random seed. A positive int number if random_state
                is not None else None, default = None.
            regularize: regularize. A str value in {"l1", "l2"} if regularize
                is not None else None, default = None.
            save_best_model: if save the best model params as the final model.
                A bool value, defalut = True

        Raises:
            AssertionError: Some parameters do not match.
        """
        (
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            save_best_model
        ) = self._init_validation(
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            save_best_model
        )
        self.random_state = random_state
        if isinstance(self.random_state, int):
            tf.random.set_seed(self.random_state)

        loss_func_dict = {"mse": keras.losses.mean_squared_error}
        self._loss_func = loss_func_dict[loss]
        metric_dict = {"mse": keras.metrics.MeanSquaredError}
        self._metric = metric_dict[loss]()

        self.alpha = alpha

        self.intercept_ = None
        if fit_intercept:
            self.intercept_ = tf.Variable(tf.random.normal([]))

        self.save_param_list = save_param_list
        self.coef_list = []
        self.intercept_list = []

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patient = patient
        self.toc = toc

        self.regularize = regularize
        self._regularizer = lambda: 0
        if self.regularize:
            reg_dict = {"l1": keras.regularizers.l1(l1=self.alpha), "l2": keras.regularizers.l2(l2=self.alpha)}
            self._regularizer = reg_dict[self.regularize]

        self.save_best_model = save_best_model

        self._X_train = None
        self._y_train = None

        self.coef_ = None

        self._optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.best_loss = tf.constant(float("inf")).numpy()
        self.best_coef_ = None
        self.best_intercept_ = None
        self.train_loss = []
        self.valid_loss = []

        self.n_iter = 0

    def fit(self, X_train, y_train, validation=None):
        """train model methed.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape,
                data type must be continuous value type.
            y_train: A np.ndarray array of (n_samples, ) shape, data type
                must be continuous value type.
            validation: the validation data for validate the model. A tuple
                like (X_valid, y_valid) , the shape of X_valid and y_valid is
                like X_train and y_train. Default = None.

        Returns:
            return self object.
        """
        self._X_train, self._y_train = self._fit_validation(X_train, y_train)
        X_train, y_train = self._X_train, self._y_train
        if self.coef_ is None:
            self.coef_ = tf.Variable(tf.random.normal(shape=[self._X_train.shape[1], 1]))
        X_valid, y_valid, X_train, y_train = self._validation_valid(validation, X_train, y_train)
        steps_per_epoch = self._X_train.shape[0] // self.batch_size
        if self.early_stopping:
            current_patient = 0
            last_val_loss = 0
        for epoch in range(self.epochs):
            # 1. train
            train_data = self._batch_generator(X_train, y_train)
            epoch_time = 0
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self._metric.reset_states()
            for step, (X_train_batch, y_train_batch) in enumerate(train_data):
                start = time.time()
                self._fit_step(X_train_batch, y_train_batch)
                epoch_time, mean_step_time, train_loss = self._step_print(steps_per_epoch, step, epoch_time, start)

            # 2. valid
            val_loss = self._epoch_valid_and_print(X_valid, y_valid, epoch_time, mean_step_time, steps_per_epoch)

            # 3. save train process
            self._save_train_process(val_loss, train_loss)

            self.n_iter += 1

            # 4. early stopping
            if self.early_stopping:
                if epoch != 0:
                    if last_val_loss - val_loss < self.toc:
                        current_patient += 1
                    else:
                        current_patient = 0
                    if current_patient >= self.patient:
                        break
                last_val_loss = val_loss

        if self.save_best_model:
            self._save_best_params()
        return self

    def _save_best_params(self):
        self.coef_ = copy.deepcopy(self.best_coef_)
        if self.intercept_ is not None:
            self.intercept_ = copy.deepcopy(self.best_intercept_)

    def _save_train_process(self, val_loss, train_loss):
        if val_loss < self.best_loss:
            self.best_loss = copy.deepcopy(val_loss)
            self.best_coef_ = copy.deepcopy(self.coef_)
            if self.intercept_ is not None:
                self.best_intercept_ = copy.deepcopy(self.intercept_)
        if self.save_param_list:
            self.coef_list.append(self.coef_.numpy().reshape([-1]))
            if self.intercept_ is not None:
                self.intercept_list.append(self.intercept_.numpy())
            self.train_loss.append(train_loss)
            self.valid_loss.append(val_loss)

    def _validation_valid(self, validation, X_train, y_train):
        if validation is None:
            n_samples = int(self._X_train.shape[0] * 0.8)
            idx = tf.random.shuffle(tf.range(self._X_train.shape[0]))
            X_train = tf.gather(self._X_train, indices=idx[:n_samples])
            y_train = tf.gather(self._y_train, indices=idx[:n_samples])
            X_valid = tf.gather(self._X_train, indices=idx[n_samples:])
            y_valid = tf.gather(self._y_train, indices=idx[n_samples:])
        else:
            X_valid, y_valid = self._fit_validation(*validation)
        return X_valid, y_valid, X_train, y_train

    @tf.function
    def _call(self, X):
        y = tf.matmul(X, self.coef_)
        if self.intercept_ is not None:
            y = tf.add(y, self.intercept_)
        return tf.reshape(y, shape=[-1])

    def _batch_generator(self, X, y, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self._X_train.shape[0])
        dataset = dataset.batch(self.batch_size)
        return dataset

    def _init_validation(
            self,
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            save_best_model,
    ):
        assert loss in {"mse"}
        assert isinstance(alpha, (int, float)) and 0 < alpha
        assert isinstance(fit_intercept, bool)
        assert isinstance(save_param_list, bool)
        assert isinstance(learning_rate, (int, float)) and 0 < learning_rate <= 1.0
        assert isinstance(epochs, int) and epochs >= 1
        assert isinstance(batch_size, int) and batch_size >= 1
        assert isinstance(early_stopping, bool)
        assert isinstance(patient, int) and patient >= 2
        assert isinstance(toc, (int, float)) and 0 < toc
        assert isinstance(random_state, (type(None), int))
        if isinstance(random_state, int):
            assert random_state >= 0
        assert regularize in {"l2", "l1", None}
        assert isinstance(save_best_model, bool)
        return (
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            save_best_model
        )

    def _fit_step(self, X_train_batch, y_train_batch):
        # 1.open a tape and calculate loss under the tape
        with tf.GradientTape() as tape:
            y_pred_batch = self._call(X_train_batch)
            loss = self._loss_func(y_train_batch, y_pred_batch)
            if self.regularize:
                loss += self._regularizer(self.coef_)
        if self.intercept_ is not None:
            # 2.use tape to calculate gradients by loss
            coef_grad, intercept_grad = tape.gradient(
                loss, [self.coef_, self.intercept_]
            )
            # 3.use optimizer to update params by gradients
            # self.coef_.assign_sub(coef_grad * self.learning_rate)
            # self.intercept_.assign_sub(intercept_grad * self.learning_rate)
            self._optimizer.apply_gradients(
                [(coef_grad, self.coef_), (intercept_grad, self.intercept_)]
            )
        else:
            coef_grad = tape.gradient(loss, self.coef_)
            self._optimizer.apply_gradients([(coef_grad, self.coef_)])
        # 4.use metric to calculate the mean loss for output
        self._metric(y_train_batch, y_pred_batch)

    def _step_print(self, steps_per_epoch, step, epoch_time, start):
        steps_str_len = len(str(steps_per_epoch))
        done_count = int((step + 1) / steps_per_epoch * 30)
        done_str = "=" * done_count
        to_do_str = "." * (30 - 1 - done_count)
        end = time.time()
        step_time = end - start
        epoch_time += step_time
        mean_step_time = epoch_time / (step + 1)
        remain_time = (steps_per_epoch - (step + 1)) * mean_step_time
        remain_time = sec2time(remain_time)
        print(
            f"\r{step + 1:{steps_str_len}}/{steps_per_epoch} [{done_str}>{to_do_str}] - ETA: {remain_time} "
            f"- loss: {self._metric.result().numpy():.4f}",
            end="",
        )
        return epoch_time, mean_step_time, self._metric.result().numpy()

    def _epoch_valid_and_print(self, X_valid, y_valid, epoch_time, mean_step_time, steps_per_epoch):
        valid_data = self._batch_generator(X_valid, y_valid, shuffle=False)
        train_mean_loss = self._metric.result().numpy()
        self._metric.reset_states()
        for valid_step, (X_valid_batch, y_valid_batch) in enumerate(valid_data):
            y_valid_pred = self._call(X_valid_batch)
            self._metric(y_valid_batch, y_valid_pred)
        epoch_time = sec2time(epoch_time)
        mean_step_time = sec2time(mean_step_time)
        print(
            f"\r{steps_per_epoch}/{steps_per_epoch} [{'=' * 30}] - {epoch_time} {mean_step_time}/step - "
            f"loss: {train_mean_loss:.4f} - val_loss: {self._metric.result().numpy():.4f}"
        )
        return self._metric.result().numpy()


class TorchSGDRegressor(TorchBaseLinearRegressor):
    """Linear SGD regressor class with tensorflow, explicitly inherits
        from TFBaseLinearRegressor already.

    Attributes:
        _X_train:feature data for training. A torch.Tensor matrix of (n_samples,
            n_features) shape, data type must be continuous value type.
        _y_train:label data for training. A torch.Tensor array of (n_samples, ) shape,
            data type must be continuous value.
        coef_: coef of linear regressor. A torch.Tensor matrix of (n_features, 1) shape.
        intercept_: intercept of regressor. A torch.Tensor integer if intercept_ is
            not None else None.
        alpha: the regularize rate. A float number and must be greater than 0,
            default = 0.001.
        save_param_list: if save param of the train process. A bool value, default = True.
        coef_list: list of coef param from the train process,
            every coef is a np.ndarray of (n_features, ) shape.
        intercept_list: list of intercept param from the train process,
            every intercept is a np.ndarray float number.
        learning_rate: learning rate. A positive float number, default = 0.001.
        epochs: epochs. A positive int number, default = 10.
        batch_size: batch size. A positive int number, default = 32.
        early_stopping: if early stopping when loss don't reduce again. A bool value,
            default = True.
        patient: Number of epochs that do not reduce loss continuously,
            patient only takes effect when early_stopping is True.
            A positive int number, default = 5.
        toc: The threshold that symbolizes loss no longer decreases,
            toc only takes effect when early_stopping is True.
            A float number, default = 0.001
        random_state: random seed. A positive int number if random_state
            is not None else None, default = None.
        regularize: regularize. A str value in {"l1", "l2"} if regularize
            is not None else None, default = None.
        best_loss: best loss of the train process. A np.ndarray float number.
        best_coef: best coef of the train process. A torch.Tensor array of
            (n_features, 1) shape.
        best_intercept_: best intercept of the train process. A torch.Tensor number.
        train_loss: list of train loss from the train process.
            every loss is a np.ndarray float number.
        valid_loss: list of valid loss from the train process.
            every loss is a np.ndarray float number.
        n_iter: the actual iteration of train process. A int number, initial = 0.
        save_best_model: if save the best model params as the final model.
                A bool value, defalut = True.
    """

    def __init__(
            self,
            loss="mse",
            alpha=0.001,
            fit_intercept=True,
            save_param_list=True,
            learning_rate=0.001,
            epochs=10,
            batch_size=32,
            early_stopping=True,
            patient=5,
            toc=0.001,
            random_state=None,
            regularize=None,
            save_best_model=True
    ):
        """TorchSGDRegressor initial method.

        Args:
            loss: A str in {"mse"}, default = "mse"
            alpha: the regularize rate. A float number and must be greater
                than 0, default = 0.001.
            fit_intercept: if fit intercept. A bool value, default = True.
            save_param_list: if save param of the train process. A bool value,
                default = True.
            learning_rate: learning rate. A positive float number, default = 0.001.
            epochs: epochs. A positive int number, default = 10.
            batch_size: batch size. A positive int number, default = 32.
            early_stopping: if early stopping when loss don't reduce again.
                A bool value, default = True
            patient: Number of epochs that do not reduce loss continuously,
                patient only takes effect when early_stopping is True.
                A positive int number, default = 5.
            toc: The threshold that symbolizes loss no longer decreases,
                toc only takes effect when early_stopping is True.
                A float number, default = 0.001
            random_state: random seed. A positive int number if random_state
                is not None else None, default = None.
            regularize: regularize. A str value in {"l1", "l2"} if regularize
                is not None else None, default = None.
            save_best_model: if save the best model params as the final model.
                A bool value, defalut = True

        Raises:
            AssertionError: Some parameters do not match.
        """
        (
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            save_best_model
        ) = self._init_validation(
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            save_best_model
        )
        self.random_state = random_state
        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)

        loss_func_dict = {"mse": F.mse_loss}
        self._loss_func = loss_func_dict[loss]

        self.alpha = alpha

        self.intercept_ = None
        if fit_intercept:
            self.intercept_ = torch.normal(mean=0.0, std=1.0, size=[]).requires_grad_()

        self.save_param_list = save_param_list
        self.coef_list = []
        self.intercept_list = []

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patient = patient
        self.toc = toc

        self.regularize = regularize
        self._regularizer = lambda: 0
        if self.regularize:
            reg_dict = {"l1": self._l1_term, "l2": self._l2_term}
            self._regularizer = reg_dict[self.regularize]

        self.save_best_model = save_best_model

        self._X_train = None
        self._y_train = None

        self.coef_ = None

        self._optimizer = None

        self.best_loss = torch.tensor(float("inf")).item()
        self.best_coef_ = None
        self.best_intercept_ = None
        self.train_loss = []
        self.valid_loss = []

        self.n_iter = 0

    def _l1_term(self, w):
        return self.alpha * torch.sum(torch.abs(w))

    def _l2_term(self, w):
        return self.alpha * torch.sum(torch.square(w))

    def fit(self, X_train, y_train, validation=None):
        """train model methed.

        Args:
            X_train: A np.ndarray matrix of (n_samples, n_features) shape,
                data type must be continuous value type.
            y_train: A np.ndarray array of (n_samples, ) shape, data type
                must be continuous value type.
            validation: the validation data for validate the model. A tuple
                like (X_valid, y_valid) , the shape of X_valid and y_valid is
                like X_train and y_train. Default = None.

        Returns:
            return self object.
        """
        self._X_train, self._y_train = self._fit_validation(X_train, y_train)
        X_train, y_train = self._X_train, self._y_train
        if self.coef_ is None:
            self.coef_ = torch.normal(mean=0.0, std=1.0, size=(self._X_train.shape[1],)).requires_grad_()
        if self._optimizer is None:
            params = [self.coef_] if self.intercept_ is None else [self.coef_, self.intercept_]
            self._optimizer = torch.optim.SGD(params=params, lr=self.learning_rate)
        train_dataset, valid_dataset = self._validation_valid(validation, X_train, y_train)

        steps_per_epoch = self._X_train.shape[0] // self.batch_size
        if self.early_stopping:
            current_patient = 0
            last_val_loss = 0

        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        for epoch in range(self.epochs):
            # 1. train
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            epoch_time = 0
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_sum_loss = 0
            for step, (X_train_batch, y_train_batch) in enumerate(train_loader):
                start = time.time()
                train_sum_loss = self._fit_step(X_train_batch, y_train_batch, train_sum_loss)
                train_mean_loss = train_sum_loss / (step + 1)
                epoch_time, mean_step_time = self._step_print(steps_per_epoch, step, epoch_time, start, train_mean_loss)

            # 2. valid
            val_loss = self._epoch_valid_and_print(epoch_time, mean_step_time, steps_per_epoch, train_mean_loss,
                                                   valid_loader)

            # 3. save train process
            self._save_train_process(val_loss, train_mean_loss)

            self.n_iter += 1

            # 4. early stopping
            if self.early_stopping:
                if epoch != 0:
                    if last_val_loss - val_loss < self.toc:
                        current_patient += 1
                    else:
                        current_patient = 0
                    if current_patient >= self.patient:
                        break
                last_val_loss = val_loss

        if self.save_best_model:
            self._save_best_params()
        return self

    def _save_best_params(self):
        self.coef_ = copy.deepcopy(self.best_coef_)
        if self.intercept_ is not None:
            self.intercept_ = copy.deepcopy(self.best_intercept_)

    def _save_train_process(self, val_loss, train_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss.item()
            self.best_coef_ = copy.deepcopy(self.coef_)
            if self.intercept_ is not None:
                self.best_intercept_ = copy.deepcopy(self.intercept_)
        if self.save_param_list:
            self.coef_list.append(copy.deepcopy(self.coef_.detach().numpy()))
            if self.intercept_ is not None:
                self.intercept_list.append(copy.deepcopy(self.intercept_.item()))
            self.train_loss.append(train_loss)
            self.valid_loss.append(val_loss)

    def _validation_valid(self, validation, X_train, y_train):
        dataset = TorchBaseDataset(X_train, y_train)
        if validation is None:
            n_samples = int(self._X_train.shape[0] * 0.8)
            train_dataset, valid_dataset = random_split(
                dataset=dataset, lengths=[n_samples, self._X_train.shape[0] - n_samples])
        else:
            X_valid, y_valid = self._fit_validation(*validation)
            train_dataset = dataset
            valid_dataset = TorchBaseDataset(X_valid, y_valid)
        return train_dataset, valid_dataset

    def _call(self, X):
        y = torch.matmul(X, self.coef_)
        if self.intercept_ is not None:
            y = torch.add(y, self.intercept_)
        return y

    def _init_validation(
            self,
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            save_best_model,
    ):
        assert loss in {"mse"}
        assert isinstance(alpha, (int, float)) and 0 < alpha
        assert isinstance(fit_intercept, bool)
        assert isinstance(save_param_list, bool)
        assert isinstance(learning_rate, float) and 0 < learning_rate <= 1.0
        assert isinstance(epochs, int) and epochs >= 1
        assert isinstance(batch_size, int) and batch_size >= 1
        assert isinstance(early_stopping, bool)
        assert isinstance(patient, int) and patient >= 2
        assert isinstance(toc, (int, float)) and 0 < toc
        assert isinstance(random_state, (type(None), int))
        if isinstance(random_state, int):
            assert random_state >= 0
        assert regularize in {"l2", "l1", None}
        assert isinstance(save_best_model, bool)
        return (
            loss,
            alpha,
            fit_intercept,
            save_param_list,
            learning_rate,
            epochs,
            batch_size,
            early_stopping,
            patient,
            toc,
            random_state,
            regularize,
            save_best_model
        )

    def _fit_step(self, X_train_batch, y_train_batch, train_sum_loss):
        # 1.calculate loss
        y_pred_batch = self._call(X_train_batch)
        loss = self._loss_func(y_train_batch, y_pred_batch)
        if self.regularize:
            loss += self._regularizer(self.coef_)

        # optimizer clean gradient
        self._optimizer.zero_grad()
        # 2.calculate gradients by loss
        loss.backward()
        # 3.use optimizer to update params by gradients
        self._optimizer.step()

        # 4.use metric to calculate the mean loss for output
        train_sum_loss += loss
        return train_sum_loss

    def _step_print(self, steps_per_epoch, step, epoch_time, start, train_mean_loss):
        steps_str_len = len(str(steps_per_epoch))
        done_count = int((step + 1) / steps_per_epoch * 30)
        done_str = "=" * done_count
        to_do_str = "." * (30 - 1 - done_count)
        end = time.time()
        step_time = end - start
        epoch_time += step_time
        mean_step_time = epoch_time / (step + 1)
        remain_time = (steps_per_epoch - (step + 1)) * mean_step_time
        remain_time = sec2time(remain_time)
        print(
            f"\r{step + 1:{steps_str_len}}/{steps_per_epoch} [{done_str}>{to_do_str}] - ETA: {remain_time} - loss: {train_mean_loss:.4f}",
            end="",
        )
        return epoch_time, mean_step_time

    def _epoch_valid_and_print(self, epoch_time, mean_step_time, steps_per_epoch, train_mean_loss, valid_loader):
        valid_mean_loss = 0
        for valid_step, (X_valid_batch, y_valid_batch) in enumerate(valid_loader):
            y_valid_pred = self._call(X_valid_batch)
            loss = self._loss_func(y_valid_batch, y_valid_pred)
            valid_mean_loss += loss
        valid_mean_loss /= (valid_step + 1)
        epoch_time = sec2time(epoch_time)
        mean_step_time = sec2time(mean_step_time)
        print(
            f"\r{steps_per_epoch}/{steps_per_epoch} [{'=' * 30}] - {epoch_time} {mean_step_time}/step - loss: {train_mean_loss:.4f} - val_loss: {valid_mean_loss:.4f}"
        )
        return valid_mean_loss
