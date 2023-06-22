from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem

def generate_dataset(n, d, k, variance)-> Tuple[np.ndarray, np.ndarray]:
    weight = np.zeros(d)
    for j in range(k): weight[j] = (j+1) / k

    np.random.seed()
    X = np.random.normal(size = (n,d))

    errors = np.random.normal(0,1,n)

    y = np.dot(weight.T, X.T) + errors.T
    return (X, y, weight)

@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    return 2*np.sum(np.power(X,2),axis=0)
    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. R espresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    # raise NotImplementedError("Your Code Goes Here")
    n = X.shape[0]
    d = X.shape[1]
    
    # Calculate ak and b of Algorithm 1

    b = 1/n * np.sum(y - np.dot(X, weight))
    if (len(weight.shape) == 1):
        weight = np.reshape(np.copy(weight), (weight.shape[0], -1))
    
    # print("shape of X is :", X.shape)
    w_new = np.zeros(weight.shape)
    
    for k in range(d):
        # ak is precalculated
        # c = 2*np.dot(X[:, k], y - (b + np.dot(weight.T, X.T) - weight[k, 0]*X[:, k]))
        prod_second_term = np.sum(X[:, k] * (y - b))
        prod_first_term =  -1 * np.sum(np.multiply(X[:, [k]], (np.matmul(X, weight) - np.multiply(X[:,[k]], weight[k, 0]))))
        c = 2 * (prod_second_term + prod_first_term)

        if c < -1*_lambda:
            w_new[k, 0] = (c + _lambda) / a[k]
        elif c > _lambda:
            w_new[k, 0] = (c - _lambda) / a[k]
        else:
            w_new[k, 0] = 0
        
        weight[k, 0] = w_new[k, 0]

    return w_new.reshape(w_new.shape[0]), b
    return w_new, b

@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    regularizer = _lambda * np.sum(np.abs(weight))

    return np.sum((np.dot(X, weight) + bias - y)**2) + regularizer


@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float .

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    
    d = X.shape[1]
    prev_w = np.ones((d,1))
    if start_weight is None:
        start_weight = np.zeros(d)
    
    w = start_weight
    a = precalculate_a(X)

    while True:
        prev_w = np.copy(w)
        w,bias = step(X,y,w,a,_lambda)
        if((convergence_criterion(w,prev_w,convergence_delta))):
            break

    return w,bias

@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    if(np.max(np.abs(weight-old_w))<convergence_delta):
        return True
    return False
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    # A1 - Generate a dataset using this model with n = 500, d = 1000, k = 100, and σ = 1. 
    n = 500
    d = 1000
    k = 100
    variance = 1
    (X, y, weight) = generate_dataset(n, d, k, variance)

    '''
    lam_max = 2*np.max(np.abs(np.dot(y.T - np.mean(y), X)))
    lam_ratio = 2
    delta = 1e-4 # threshold to stop iteration (search for better w)

    current_lam = lam_max
    lam_vals = []
    prev_w = weight

    
    W = [] # we will store the resulting w for each lambda as a column in this matrix
    
    for count in range(0,50):
        
        if count%5 == 0:
            print("Running ", count)

        # Initialize current round of coordinate descent to w found in last round
        (w_new, b) = train(X, y, current_lam, delta, prev_w) 
        
        W.append(np.count_nonzero(w_new)) 
        # print("new weight is : ", w_new)
        print("lambda is : ", current_lam)
        lam_vals.append(current_lam)
        current_lam = current_lam / lam_ratio
        prev_w = np.copy(w_new)
    
    plt.figure(1)
    plt.plot(lam_vals, W)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Nonzero Coefficients in w')
    plt.title('7a: Nonzero Coefficients versus Lambda')
    plt.show()
    '''

    
    X, y, weight = generate_dataset(n, d, k, variance)
    # print(weight)
    # exit(0)
    maxLambda = 2*np.max(np.abs(np.dot(y.T - np.mean(y), X)))
    
    zeros = []
    lambda_list = []
    newLambda = maxLambda

    t_weight = np.copy(weight)
    corr_non_zero_list = []
    incorr_non_zero_list = []
    for count in range(0, 20):
        
        new_weight, bias = train(X, y, newLambda, 5e-4, weight)
        # print("new weight is : ", new_weight)
        # print("lambda is : ", newLambda)
        print("count is : ", count)
        number_non_zeros = np.count_nonzero(new_weight)
        
        lambda_list.append(newLambda)
        newLambda = newLambda/2
        t_new_weight = np.copy(new_weight)
        # t_new_weight = np.reshape(t_new_weight, (X.shape[1], -1))

        incorr_non_zeros = np.sum((t_weight == 0)& (t_new_weight != 0))
        # print("incorrent non zeros : ", incorr_non_zeros)
        if number_non_zeros == 0:
            incorr_non_zero_list.append(0)
        else:
            incorr_non_zero_list.append(incorr_non_zeros/number_non_zeros)
        # print("t weight is ; ", t_weight)
        corr_non_zeros = np.sum((t_weight != 0)& (t_new_weight != 0))
        # print("correct non zeros : ", corr_non_zeros)
        
        corr_non_zero_list.append(corr_non_zeros/100)
        
        #weight=weightme
        zeros.append(number_non_zeros)

        if (np.count_nonzero(new_weight==0)<5):
            break
    # carl.append([weight,np.count_nonzero(weight==0),maxLambda])
    

    zeros = np.asarray(zeros)
    # print("zeros are : ", zeros)
    lambda_list = np.asarray(lambda_list)
    # print("lambda list is : ", lambda_list)
    plt.plot(lambda_list, zeros)
    # plt.plot(zeros, lambda_list)
    plt.xscale('log')

    # plt.show()
    plt.xlabel('log(lambda)')
    plt.ylabel('No. of non-zeros')
    plt.savefig("lambda_vs_non_zero.png")
    plt.show()

    incorr_non_zero_list = np.asarray(incorr_non_zero_list)
    corr_non_zero_list = np.asarray(corr_non_zero_list)
    plt.plot(incorr_non_zero_list, corr_non_zero_list)
    plt.xlabel('False Detection Rate')
    plt.ylabel('True Positive Rate')
    # plt.plot(zeros, lambda_list)
    # plt.xscale('log')

    # plt.show()
    plt.savefig("FDR_TPR.png")

    
    # raise NotImplementedError("Your Code Goes Here")

if __name__ == "__main__":
    main()

