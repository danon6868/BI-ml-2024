import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    y_true = y_true.squeeze()
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = None
        print("Precision contains 0 division")
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = None
        print("Recall contains 0 division")
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = None
        print("F1 contains 0 division")
    try:
        accuracy = (tp + tn) / (tp + tn + fn + fp)
    except ZeroDivisionError:
        accuracy = None
        print("Accuracy calculations contain zero division")

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    y_true = y_true.astype(int).squeeze()
    classes = np.unique(y_true)
    for i in range(len(classes)):
        tp += np.sum((y_pred == classes[i]) & (y_true == classes[i]))
        tn += np.sum((y_pred != classes[i]) & (y_true != classes[i]))
        fp += np.sum((y_pred == classes[i]) & (y_true != classes[i]))
        fn += np.sum((y_pred != classes[i]) & (y_true == classes[i]))
    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        accuracy = None
        print("Accuracy calculations contain zero division")

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    y_mean = np.mean(y_true)
    total_sum_squares = np.sum((y_true - y_mean) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (residual_sum_squares / total_sum_squares)

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """

    return ((y_true - y_pred) ** 2).mean()


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """

    return np.abs((y_true - y_pred)).mean()