import numpy as np


def load_linear_data(point_count=100, min_=0., max_=10., w=0.5493, b=1.1973, random_state=None, scale=1.0, loc=0.0):
    """load linear data

    Args:
        point_count: a integer number, default = 100.
        min_: bottom range of x data, a float number, default = 0.0.
        max_: top range of x data, a float number, default = 10.0.
        w: the coef of linear, a float number, default = 0.5493.
        b: the intercept of linear, a float number, default = 1.1973.
        random_state: random seed, a int number, default = None.
        scale: noise's scale. A float number, default = 1.0.
        loc: noise's loc. A float number, default = 0.0

    Returns:
        A tuple. (x, y). the shape of x is (point_count, 1), the shape
            of y is (point_count, ).

    Raises:
        AssertionError: random_state is not a integer.
    """
    if random_state is not None:
        assert isinstance(random_state, int)
        np.random.seed(random_state)

    x = np.random.uniform(min_, max_, point_count)
    noise = np.random.normal(scale=scale, loc=loc, size=[point_count])
    y = w * x + b + noise
    return x.reshape([-1, 1]), y


def load_data_from_func(func=lambda X_data: 2.1084 * np.square(X_data) - 0.1932 * X_data + 10.813,
                        x_min=0, x_max=10, n_samples=500, loc=0, scale=1, random_state=None):
    """load point data from a function

    Args:
        func: Function for creating data.A Function object, default = lambda X_data:
            2.1084 *  np.square(X_data) - 0.1932 * X_data + 10.813.
        x_min: min value of x. A number, x_min must be less ther x_max, default = 0.
        x_max: max value of x. A number, x_max must be greater ther x_min, default = 0.
        n_samples: sample count. A int number, default = 500.
        loc: loc of noise's destribution. A float number, default = 0.
        scale: scale of noise's destribution. A float number, default = 1.
        random_state: random seed. A positive int number, default = None.

    Returns:
        A tuple of x and y. x's shape is (n_samples, 1), y's shape is (n_samples, )
    """
    if random_state is not None and isinstance(random_state, int):
        np.random.seed(random_state)
    x = np.random.uniform(x_min, x_max, n_samples)
    y = func(x)
    noise = np.random.normal(loc=loc, scale=scale, size=n_samples)
    y += noise
    return x.reshape([-1, 1]), y