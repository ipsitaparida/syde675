if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_dataset, problem

def calc_lambda(X, y):
    
    return 2*np.max(np.abs(np.dot(y.T - np.mean(y), X)))

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    y_train = df_train["ViolentCrimesPerPop"].values
    X_train = df_train.drop("ViolentCrimesPerPop", axis=1).values
    y_test = df_test["ViolentCrimesPerPop"].values
    X_test = df_test.drop("ViolentCrimesPerPop", axis=1).values
    
    lam_max = calc_lambda(X_train, y_train)
    lam_ratio = 2
    delta = 1e-4 # threshold to stop iteration (search for better w)

    current_lam = lam_max
    lam_vals = []
    prev_w = None

    
    W = None # we will store the resulting w for each lambda as a column in this matrix
    B = []
    for count in range(0,50):
        
        if count%5 == 0:
            print("Running ", count)

        # Initialize current round of coordinate descent to w found in last round
        (w_new, b) = train(X_train, y_train, current_lam, delta, prev_w) 
        
        if W is None:
            W = np.expand_dims(w_new, axis=1)
        else:
            W = np.concatenate((W, np.expand_dims(w_new, axis=1)), axis=1)
        B.append(b)
        lam_vals.append(current_lam)
        current_lam = current_lam / lam_ratio
        prev_w = np.copy(w_new)
    

    plt.figure(1)
    plt.plot(lam_vals, np.count_nonzero(W, axis=0))
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Nonzero Coefficients in w')
    plt.title('Nonzero Coefficients versus Lambda')
    plt.savefig("8c_lambda_vs_non_zero.png")
    # plt.show()

    # Plot the regularization paths
    i1 = np.where(df_train.columns == "agePct12t29")[0] - 1
    i2 = np.where(df_train.columns == "pctWSocSec")[0] - 1
    i3 = np.where(df_train.columns == "pctUrban")[0] - 1
    i4 = np.where(df_train.columns == "agePct65up")[0] - 1
    i5 = np.where(df_train.columns == "householdsize")[0] - 1

    k = len(lam_vals)
    plt.figure(2)
    plt.plot(lam_vals, np.reshape(W[i1, :], (k, )), \
             lam_vals, np.reshape(W[i2, :], (k,)), \
             lam_vals, np.reshape(W[i3, :], (k,)), \
             lam_vals, np.reshape(W[i4, :], (k,)), \
             lam_vals, np.reshape(W[i5, :], (k,)))
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient Value')
    plt.title('Regularization Paths')
    plt.legend(["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"])
    plt.savefig("8d_reg.png")

    # Squared Error on training/test data

    y_pred_train = np.dot(W.T, X_train.T) + np.expand_dims(B, axis=1)
    SSE_train = 1/X_train.shape[0] * np.sum(np.square(y_pred_train - y_train), axis=1)
    y_pred_test = np.dot(W.T, X_test.T) + np.expand_dims(B, axis=1)
    SSE_test = 1/X_test.shape[0] * np.sum(np.square(y_pred_test - y_test), axis=1)

    plt.figure(3)
    plt.plot(lam_vals, SSE_train, lam_vals, SSE_test)
    plt.xscale('log')
    plt.legend(["Training Error", "Testing Error"])
    plt.xlabel('Lambda')
    plt.ylabel('SSE / n')
    plt.title('Squared Error as a function of Lambda')
    plt.savefig("8e_SSE.png")
    plt.show()
    
    (w30, _ ) = train(X_train, y_train, 30, delta, prev_w)

    var_names = df_train.columns[1:] # skip the first varname corresponding to response variable ViolentCrimesPerPop
    nonzero_coeffs = {w30[i]:var_names[i] for i in range(len(w30)) if not w30[i] == 0}
    max_coeff = max(list(nonzero_coeffs.keys()))
    min_coeff = min(list(nonzero_coeffs.keys())) 
    print(nonzero_coeffs)
    print("feature with largest coefficient: ", nonzero_coeffs[max_coeff], " , value: ", max_coeff)
    print("feature with smallest coefficient: ", nonzero_coeffs[min_coeff], " , value: ", min_coeff)
    
    return
    raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
