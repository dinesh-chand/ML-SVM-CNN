import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data():
    # load the data
    # return X_train, X_test, y_train, y_test
    raise NotImplementedError


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    raise NotImplementedError


def plot_metrics(metrics) -> None:
    # plot and save the results
    raise NotImplementedError