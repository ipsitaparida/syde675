from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    k = (np.multiply.outer(x_i, x_j) + 1) ** d
    return k


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    K = np.exp(-gamma * ((np.subtract.outer(x_i, x_j)**2)))
    return K


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    K = kernel_function(x, x, kernel_param)
    alpha_hat = np.linalg.solve(K + _lambda * np.eye(K.shape[0]), y)
    return alpha_hat


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    loss = []

    for i in range(num_folds):
        start, end = i * fold_size, (i+1)*fold_size

        x_train, y_train = np.append(x[:start], x[end:]), np.append(y[:start], y[end:])
        x_test, y_test = x[start: end], y[start: end]
        
        alpha_hat = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        K = kernel_function(x_train, x_test, kernel_param)

        predict = alpha_hat@K
        loss.append(np.mean((predict-y_test)**2))

    return np.mean(loss)


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    # approach - Grid search
    n = len(x)
    dists = []

    for i in range(n):
        for j in range(i+1, n):
            dists.append((x[i]-x[j])**2)

    gamma = 1 / np.median(dists)
    gamma_dist = np.random.normal(gamma, 1, 50) # size = 50

    min_loss = float("inf")
    min_lambda = None
    min_gamma = None

    lambda_list = 10 ** np.linspace(-5, -1, num=100)

    for lmb in lambda_list:
        for gamma in gamma_dist:
            loss = cross_validation(x, y, rbf_kernel, gamma, lmb, num_folds)
            if loss < min_loss:
                min_gamma = gamma
                min_loss = loss
                min_lambda = lmb
            
    return (min_lambda, min_gamma)
            
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j) for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """
    # approach - Grid search
    n = len(x)

    min_loss = float("inf")
    min_lambda = None
    min_d = None

    lambda_list = 10 ** np.linspace(-5, -1, num=100)
    poly_list = np.arange(5, 26)

    for lmb in lambda_list:
        for d in poly_list:
            loss = cross_validation(x, y, poly_kernel, d, lmb, num_folds)
            if loss < min_loss:
                min_d = d
                min_loss = loss
                min_lambda = lmb

    return (min_lambda, min_d)


@problem.tag("hw3-A", start_line=1)
def bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    bootstrap_iters: int = 300,
) -> np.ndarray:
    """Bootstrap function simulation empirical confidence interval of function class.

    For each iteration of bootstrap:
        1. Sample len(x) many of (x, y) pairs with replacement
        2. Train model on these sampled points
        3. Predict values on x_fine_grid (see provided code)

    Lastly after all iterations, calculated 5th and 95th percentiles of predictions for each point in x_fine_point and return them.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        bootstrap_iters (int, optional): [description]. Defaults to 300.

    Returns:
        np.ndarray: A (2, 100) numpy array, where each row contains 5 and 95 percentile of function prediction at corresponding point of x_fine_grid.

    Note:
        - See np.percentile function.
            It can take two percentiles at the same time, and take percentiles along specific axis.
    """
    x_fine_grid = np.linspace(0, 1, 100)
    result = None
    
    for i in range(bootstrap_iters):
        # For each attempt, sample len(x)
        idx = np.random.choice(len(x), len(x))
        x_iter = np.array([x[i] for i in idx])
        y_iter = np.array([y[i] for i in idx])

        # train, predict
        alpha = train(x_iter, y_iter, kernel_function, kernel_param, _lambda)
        K = kernel_function(x_iter, x_fine_grid, kernel_param)
        predict = (alpha@K).reshape((1, -1))

        if result is None:
            result = predict
        else:
            result = np.append(result, predict, axis=0)
    
    return np.percentile(result, [5, 95], axis=0)


@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        C. For both rbf and poly kernels, plot 5th and 95th percentiles from bootstrap using x_30, y_30 (using the same fine grid as in part B)
        D. Repeat A, B, C with x_300, y_300
        E. Compare rbf and poly kernels using bootstrap as described in the pdf. Report 5 and 95 percentiles in errors of each function.

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")

    ''' 
    Part A. find a good λ and hyperparameter. Use the value for next part of the questions
    '''
    print("------A------")
    (poly_opt_lambda, poly_opt_dim) = poly_param_search(x_30, y_30, len(x_30))
    print("A. Poly kernal best value - lambda: ", poly_opt_lambda ,", d: ", poly_opt_dim)
    (rbf_opt_lambda, rbf_opt_gamma) = rbf_param_search(x_30, y_30, len(x_30))
    print("A. RBF kernal best value - lambda: ", rbf_opt_lambda, ", gamma: ", rbf_opt_gamma)
    
    ''' 
    Part B. 
    '''
    x = np.linspace(0, 1, 100)
    true_y = f_true(x)

    poly_alpha = train(x_30, y_30, poly_kernel, poly_opt_dim, poly_opt_lambda)
    poly_K = poly_kernel(x_30, x, poly_opt_dim)
    poly_y = poly_alpha@poly_K

    RBF_alpha = train(x_30, y_30, rbf_kernel, rbf_opt_gamma, rbf_opt_lambda)
    RBF_K = rbf_kernel(x_30, x, rbf_opt_gamma)
    RBF_y = RBF_alpha@RBF_K
        
    plt.plot(x_30, y_30, "ko", label="data n=30", markersize=3)
    plt.plot(x, true_y, "b-", label="true f")
    plt.plot(x, poly_y, "g-", label="Poly Kernel")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    plt.plot(x_30, y_30, "ko", label="data n=30", markersize=3)
    plt.plot(x, true_y, "b-", label="true f")
    plt.plot(x, RBF_y, "r-", label="RBF Kernel")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

    '''
    Part C
    '''
    poly_boot = bootstrap(x_30, y_30, poly_kernel, poly_opt_dim, poly_opt_lambda, 300)
    RBF_boot = bootstrap(x_30, y_30, rbf_kernel, rbf_opt_gamma, rbf_opt_lambda, 300)

    plt.plot(x_30, y_30, "ko", label="data n=30", markersize=3)
    plt.plot(x, true_y, "b-", label="true f")
    plt.plot(x, poly_y, "g-", label="Poly Kernel")
    plt.fill_between(x, poly_boot[0], poly_boot[1], color = "b", alpha=0.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim((-10, 15))
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(x_30, y_30, "ko", label="data n=30", markersize=3)
    plt.plot(x, true_y, "b-", label="true f")
    plt.plot(x, RBF_y, "r-", label="RBF Kernel")
    plt.fill_between(x, RBF_boot[0], RBF_boot[1], color = "r", alpha=0.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

    '''
    Part D Repeat parts a and b with n = 300, but use 10-fold CV
    '''
    print("-----D-----")
    (poly_opt_lambda, poly_opt_dim) = poly_param_search(x_300, y_300, 10)
    print("D. Poly kernal best value - lambda: ", poly_opt_lambda ,", d: ", poly_opt_dim)
    (RBF_opt_lambda, RBF_opt_gamma) = rbf_param_search(x_300, y_300, 10)
    print("D. RBF kernal best value - lambda: ", RBF_opt_lambda, ", gamma: ", RBF_opt_gamma)

    poly_alpha = train(x_300, y_300, poly_kernel, poly_opt_dim, poly_opt_lambda)
    RBF_alpha = train(x_300, y_300, rbf_kernel, RBF_opt_gamma, RBF_opt_lambda)

    x = np.linspace(0, 1, 100)
    true_y = f_true(x)

    poly_K = poly_kernel(x_300, x, poly_opt_dim)
    poly_y = poly_alpha@poly_K

    RBF_K = rbf_kernel(x_300, x, RBF_opt_gamma)
    RBF_y = RBF_alpha@RBF_K

    poly_boot = bootstrap(x_300, y_300, poly_kernel, poly_opt_dim, poly_opt_lambda, 300)
    RBF_boot = bootstrap(x_300, y_300, rbf_kernel, RBF_opt_gamma, RBF_opt_lambda, 300)

    plt.plot(x_300, y_300, "ko", label="data n=300", markersize=3)
    plt.plot(x, true_y, "b-", label="true f")
    plt.plot(x, poly_y, "g-", label="Poly Kernel")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(x_300, y_300, "ko", label="data n=300", markersize=3)
    plt.plot(x, true_y, "b-", label="true f")
    plt.plot(x, RBF_y, "r-", label="RBF Kernel")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(x_300, y_300, "ko", label="data n=300", markersize=3)
    plt.plot(x, true_y, "b-", label="true f")
    plt.plot(x, poly_y, "g-", label="Poly Kernel")
    plt.fill_between(x, poly_boot[0], poly_boot[1], color = "b", alpha=0.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(x_300, y_300, "ko", label="data n=300", markersize=3)
    plt.plot(x, true_y, "b-", label="true f")
    plt.plot(x, RBF_y, "r-", label="RBF Kernel")
    plt.fill_between(x, RBF_boot[0], RBF_boot[1], color = "r", alpha=0.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

    '''
    Part e
    '''
    print("------E-----")
    result = []
        
    for j in range(300):
        iter = np.random.choice(len(x_1000), 1000)
        x_iter = np.array([x_1000[i] for i in iter])
        y_iter = np.array([y_1000[i] for i in iter])

        poly_K = poly_kernel(x_300, x_iter, poly_opt_dim)
        RBF_K = rbf_kernel(x_300, x_iter, RBF_opt_gamma)

        poly_predict = (poly_alpha@poly_K).reshape((1,-1))
        RBF_predict = (RBF_alpha@RBF_K).reshape((1,-1))

        result.append(np.mean((poly_predict-y_iter)**2 - (RBF_predict-y_iter)**2))

    p5, p95 = np.percentile(np.array(result), [5, 95])
    
    print(p5, p95)

if __name__ == "__main__":
    main()
