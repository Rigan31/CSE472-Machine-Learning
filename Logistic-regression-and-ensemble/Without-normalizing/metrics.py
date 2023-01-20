"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
import numpy as np

def accuracy(y_true, y_pred):
    # find difference between y_true and y_pred
    # and divide by total number of elements
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

def precision_score(y_true, y_pred):
    # precision = TP / (TP + FP)
    # TP = True Positive
    # FP = False Positive
    # FN = False Negative
    # TN = True Negative
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    FP = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    precision = TP / (TP + FP)
    return precision

def recall_score(y_true, y_pred):
    # recall = TP / (TP + FN)
    # TP = True Positive
    # FP = False Positive
    # FN = False Negative
    # TN = True Negative
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    recall = TP / (TP + FN)
    return recall


def f1_score(y_true, y_pred):
    # f1 = 2 * (precision * recall) / (precision + recall)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
