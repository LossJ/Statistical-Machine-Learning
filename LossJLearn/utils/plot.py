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
