import numpy as np


def clouds(num_points=100):
    """
    Generate synthetic data for a two-class classification problem.

    Parameters:
    -----------
        `num_points` (`int`): Number of points to generate for each class. (default=100)

    Returns:
    --------
        `tuple`: A tuple of two lists, `X` and `y`, where `X` is a list of points and `y` is a list of labels for each point.
    """
    centers = [(1, 1), (-1, -1)]
    spreads = [0.5, 0.7]
    labels = [-1, 1]

    X = []
    y = []

    for center, spread, label in zip(centers, spreads, labels):
        X += np.random.multivariate_normal(
            center, spread * np.identity(2), num_points
        ).tolist()
        y += [label] * num_points

    return X, y


def circle(num_points=250):
    """
    Generates a synthetic dataset of points distributed in a circle.

    Parameters:
    ------------

        `num_points`: `int`, the number of points in the dataset.

    Returns:
    --------

        `tuple`: (`points`, `labels`), where points is a 2D array of size `(num_points, 2)` and labels is a 1D array of size `num_points`.
    """
    points = 1 - 2 * np.random.random((num_points, 2))
    radius = 0.6
    labels = [1 if np.linalg.norm(point) > radius else -1 for point in points]
    return points, labels
