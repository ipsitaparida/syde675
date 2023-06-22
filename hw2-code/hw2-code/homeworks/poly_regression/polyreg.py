"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from operator import mod
from typing import Tuple
from sklearn import preprocessing

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        #raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        arr = X
        for i in range(1, degree):
            arr = np.c_[arr, X ** (i+1)]
        return arr
        raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        poly_arr = self.polyfeatures(X, self.degree)
        self.mean = np.mean(poly_arr, 0)
        self.std = np.std(poly_arr, 0)
        poly_arr = (poly_arr - self.mean) / self.std
        
        n = len(X)
        # add 1s column
        X_ = np.c_[np.ones([n, 1]), poly_arr]

        n, d = X_.shape

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d)
        # reg_matrix[0, 0] = 0
        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.inv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)

        # raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        poly_arr = self.polyfeatures(X, self.degree)
        poly_arr = (poly_arr - self.mean) / self.std
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), poly_arr]

        # predict
        return X_.dot(self.weight)

        raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    mse = np.square(np.subtract(a,b)).mean()
    
    return mse


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays

    model = PolynomialRegression(degree= degree, reg_lambda= reg_lambda)
 
    for i in range(1, n):
        model.fit(Xtrain[0:(i + 1)], Ytrain[0:(i + 1)])
        Ypredict_train = model.predict(Xtrain[0:(i + 1)])
        Ypredict_test = model.predict(Xtest[0:(i + 1)])
        
        errorTrain[i] = mean_squared_error(Ypredict_train,Ytrain[0:(i + 1)])
        errorTest[i] = mean_squared_error(Ypredict_test, Ytest[0:(i + 1)])

    return(errorTrain, errorTest)
    raise NotImplementedError("Your Code Goes Here")
