from matplotlib import pyplot as plt
import numpy as np


def show_regressor_pred(pred, y_test, y_train, x_magnification=1, y_magnification=1):
    """Plot comparison between regressor's prediction and true label and mean label from y_train.

    Args:
        pred: Prediction Label of test dataset. A np.ndarray of (sample_count,) shape.
        y_test: Labels of test dataset. A np.ndarray of (sample_count,) shape.
        y_train: Labels of train dataset. A np.ndarray of (sample_count,) shape.
        x_magnification: Magnification for X axis. A interger for float number, default = 1.
        y_magnification: Magnification for y axis. A interger for float number, default = 1
    """
    plt.figure(figsize=(18 * x_magnification, 9 * y_magnification))
    plt.plot(range(y_test.shape[0]), y_test, label="true")
    plt.plot(range(y_test.shape[0]), pred, label="prediction")
    plt.scatter(range(y_test.shape[0]), [np.mean(y_train)] * y_test.shape[0], label="mean", s=2, c="green")
    plt.xlabel("test sample index")
    plt.ylabel("label unit")
    plt.legend()
    plt.show()


def show_prediction_face_comparison(test, pred_splice, img_num=10):
    """Plot comparision between true face and prediction face.s

    Args:
        test: Complete test face image data. A np.ndarray of (sample_count, height, weight) shape.
        pred_splice: Image data spliced with half of test true face top part and the face prediction part.
            A np.ndarray of (sample_count, height, weight) shape.
        img_num: Image count want to show. A integer number, default = 10.
    """
    plt.figure(figsize=(10, img_num * 5))
    for i in range(img_num):
        plt.subplot(img_num, 2, 2 * i + 1)
        plt.title(f"true face {i+1}")
        plt.imshow(test[i], cmap="gray", )
        plt.axis('off')

        plt.subplot(img_num, 2, 2 * (i + 1))
        plt.title(f"predict face {i+1}")
        plt.imshow(pred_splice[i], cmap="gray")
        plt.axis('off')
    plt.show()


def show_linear_point(X_data, y_data, s=12):
    """Plot linear point

    Args:
         X_data: X_data with (sample_count, 1) shape.
         y_data: y_data with (sample_count, 1) shape.
    """
    plt.figure(figsize=[10, 5])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X_data, y_data, s=s)
    plt.show()


def show_regressor_linear(X_data, y_data, pred_coef, pred_intercept=0):
    """show the regression linear and data point.

    Args:
        X_data: A np.ndarray matrix of (point_count, 1) shape,
            data type must be continuous value type.
        y_data: A np.ndarray array of (point_count, ) shape.
        pred_coef: prediction coef. A np.ndarray matrix of (1, ) shape.
        pred_intercept: prediction coef. A np.ndarray integer or interge,
            default = 0.
    """
    if pred_intercept is None:
        pred_intercept = 0
    plt.figure(figsize=[10, 5])
    plt.scatter(X_data, y_data, label="true", s=12)
    pred_y = pred_coef * X_data + pred_intercept
    plt.plot(X_data, pred_y, label="pred", color="darkorange")
    plt.legend()
    plt.show()


def show_regressor_linear_sgd(X_data, y_data, coef_list, best_coef, intercept_list=None, best_intercept=None,
                              pause_second=0.001, step=1, max_iter=50):
    """show the params' change of the train process.

    Args:
        X_data: A np.ndarray matrix of (point_count, 1) shape,
            data type must be continuous value type.
        y_data: A np.ndarray array of (point_count, ) shape.
        coef_list: list with coef param( A np.ndarray array of (n_features, ) shape)
            for every epochs.
        best_coef: the best coef param. A np.ndarray array of (n_features, ) shape.
        intercept_list: list with intercept param(A np.ndarray float number)
            for every epochs. default = None.
        best_intercept: the best intercept param. A np.ndarray float number.
            default = None.
        pause_second: pause second. A positive float number, default = 0.001.
        step: epoch params update step.A positive int number.default = 1.
        max_iter: max epoch iter. A positive int number. default = 50.
    """
    from IPython import display

    y_bottom = np.min(y_data) - 1
    y_top = np.max(y_data) + 1
    x_bottom = np.min(X_data) - 1
    x_top = np.max(X_data) + 1

    if not intercept_list or intercept_list == [None] * len(intercept_list):
        intercept_list = [0] * len(coef_list)
    if not best_intercept:
        best_intercept = 0

    max_iter = max_iter * step
    if max_iter > len(coef_list):
        max_iter = len(coef_list)

    first_coef = None
    first_intercept = None
    for idx in range(0, max_iter, step):
        plt.figure(figsize=[20, 10])
        plt.ylim(y_bottom, y_top)
        plt.xlim(x_bottom, x_top)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(X_data, y_data, label="true", s=12)

        if idx == 0:
            first_coef = coef_list[idx]
            first_intercept = intercept_list[idx]
        else:
            first_pred_y = first_coef * X_data + first_intercept
            plt.plot(
                X_data,
                first_pred_y,
                linewidth=1,
                linestyle="-",
                label="first_pred",
                color="yellowgreen",
            )

        if idx == (len(coef_list) - 1) or idx + step >= max_iter:
            best_pred_y = best_coef * X_data + best_intercept
            plt.plot(X_data, best_pred_y, label="best_pred", color="firebrick")

        pred_y = coef_list[idx] * X_data + intercept_list[idx]
        plt.plot(X_data, pred_y, label="pred", color="darkorange")
        plt.legend()
        plt.show()
        plt.pause(pause_second)
        display.clear_output(wait=True)


def show_regression_line(X_data, y_data, y_pred):
    """show regression line no only linear regression.

    Args:
        X_data: A np.ndarray matrix of (point_count, 1) shape,
            data type must be continuous value type.
        y_data: A np.ndarray array of (point_count, ) shape.
        y_pred: A np.ndarray array of (point_count, ) shape.
    """
    plt.figure(figsize=[10, 5])
    plt.xlabel("x")
    plt.ylabel("y")
    if X_data.ndim == 2:
        X_data = X_data.reshape(-1)
    plt.scatter(X_data, y_data)
    idx = np.argsort(X_data)
    X_data = X_data[idx]
    y_pred = y_pred[idx]
    plt.plot(X_data, y_pred, color="darkorange")
    plt.show()


def show_regressor_loss(train_loss, valid_loss, x_magnification=1, y_magnification=1):
    """show regressor loss of the train process.

    Args:
        train_loss: list of train loss. train loss's type is np.ndarray float number.
        valid_loss: list of valid loss. valid loss's type is np.ndarray float number.
        x_magnification: magnification for x axis, a positive number, default = 1.
        y_magnification: magnification for y axis, a positive number, default = 1.
    """
    plt.figure(figsize=[10 * x_magnification, 5 * y_magnification])
    x = range(len(train_loss))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(x, train_loss, color="darkorange", label="train_loss")
    plt.plot(x, valid_loss, label="valid_loss")
    plt.legend()
    plt.show()


def show_diff_lr_params(lr_w1_list, lr_w2_list, lr_list, w1_name="coef", w2_name="intercept", true_w1=0.5493,
                        true_w2=1.1973):
    """show params' change on different learning rate.

    Args:
        lr_w1_list: the list of different w1 list(rgs.coef_list).
        lr_w2_list: the list of different w1 list(rgs.intercept_list).
        lr_list: the list the learning rate. learning rate is a number.
            e.g: [0.1, 0.01, 0.001, 0.0001].
        w1_name: the name of w1. A str value, default = "coef".
        w2_name: the name of w1. A str value, default = "intercept".
        true_w1: the true w1 value. A number, default = 0.5493.
        true_w2: the true w2 value. A number, default = 1.1973.
    """
    assert isinstance(lr_w1_list, list) and lr_w1_list
    assert isinstance(lr_w2_list, list) and lr_w2_list
    plt.style.use('ggplot')
    plt.figure(figsize=[16, 8])
    plt.xlabel(w1_name)
    plt.ylabel(w2_name)
    for w1_list, w2_list, lr in zip(lr_w1_list, lr_w2_list, lr_list):
        w1_list = [w1[0] if (isinstance(w1, np.ndarray) and w1.ndim >= 1) else w1 for w1 in w1_list]
        w2_list = [w2[0] if (isinstance(w2, np.ndarray) and w2.ndim >= 1) else w2 for w2 in w2_list]
        plt.plot(w1_list, w2_list, label=str(lr) if not isinstance(lr, str) else lr)
    plt.scatter(true_w1, true_w2, s=20, label="true params")
    plt.legend()
    plt.show()
