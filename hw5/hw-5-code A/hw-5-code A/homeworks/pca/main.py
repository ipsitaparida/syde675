from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a recontruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """
    projected = demean_data @ uk @ uk.T
    return projected


@problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """
    data_hat = reconstruct_demean(uk, demean_data)
    error = np.linalg.norm(data_hat - demean_data, axis=1)**2
    return np.mean(error)


@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of it.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,)
            2. Matrix with eigenvectors as columns with shape (d, d)
    """
    covmat = np.cov(demean_data.T, bias=True)
    return np.linalg.eig(covmat)

@problem.tag("hw4-A", start_line=2)
def main():
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues

    Part C:
        - Plot reconstruction error as a function of k (# of eigenvectors used)
            Use k from 1 to 101.
            Plot should have two lines, one for train, one for test.
        - Plot ratio of sum of eigenvalues remaining after k^th eigenvalue with respect to whole sum of eigenvalues.
            Use k from 1 to 101.

    Part D:
        - Visualize 10 first eigenvectors as 28x28 grayscale images.

    Part E:
        - For each of digits 2, 6, 7 plot original image, and images reconstruced from PCA with
            k values of 5, 15, 40, 100.
    """
    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")
    demean_xtr = x_tr - np.mean(x_tr, axis=0)

    # Part a 
    ind = [0, 1, 9, 29, 49]
    eig = calculate_eigen(demean_xtr)
    print(eig[0][ind])
    print(np.sum(eig[0]))

    # Part c
    k_list = list(range(100))
    demean_xtr = x_tr - np.mean(x_tr, axis=0).reshape(1, -1)
    demean_xte = x_test - np.mean(x_test, axis=0).reshape(1, -1)
        
    train_err = []
    test_err = []
    ratio = []

    tot_lambda = np.sum(eig[0])

    for k in tqdm(k_list):
        tre = reconstruction_error(eig[1][:, :k], demean_xtr)
        tee = reconstruction_error(eig[1][:, :k], demean_xte)
        r = 1 - np.sum(eig[0][:k])/tot_lambda

        train_err.append(tre)
        test_err.append(tee)
        ratio.append(r)
    
    plt.plot(k_list, train_err, "-b", label="Train errors")
    plt.plot(k_list, test_err, "-r", label="Test errors")
    plt.xlabel("K")
    plt.ylabel("Reconstruction errors")
    plt.legend()
    plt.show()

    plt.plot(k_list, ratio, "-g", label="Ratio")
    plt.xlabel("K")
    plt.ylabel("Ratios")
    plt.legend()
    plt.show()

    #Part d
    fig, ax = plt.subplots(2, 5)
    for i in range(10):
        ax.ravel()[i].imshow(eig[1][:, i].reshape((28,28)))
        ax.ravel()[i].axes.xaxis.set_visible(False)
        ax.ravel()[i].axes.yaxis.set_visible(False)
    fig.suptitle('First 10 eigenvectors as images')
    plt.show()

    # Plot e
    data_idx = [5, 13, 15]
    k_val = [5, 15, 40, 100]
    demean_xtr = x_tr - np.mean(x_tr, axis=0)

    fig, ax = plt.subplots(3, 5)
    for i in range(3):
        ax.ravel()[i*5].imshow(x_tr[data_idx[i], :].reshape((28,28)))
        ax.ravel()[i*5].axes.xaxis.set_visible(False)
        ax.ravel()[i*5].axes.yaxis.set_visible(False)
        for j in range(4):
            ax.ravel()[i*5+j+1].imshow(reconstruct_demean(eig[1][:, :k_val[j]], demean_xtr)[data_idx[i], :].reshape(28,28))
            ax.ravel()[i*5+j+1].axes.xaxis.set_visible(False)
            ax.ravel()[i*5+j+1].axes.yaxis.set_visible(False)
    fig.suptitle('2, 6, 7 visualization with K = 5, 15, 40, 100')
    plt.show()

    # Plot a3,d
    data_idx = 5
    k_val = [32, 64,128]
    demean_xtr = x_tr - np.mean(x_tr, axis=0)
    fig, ax = plt.subplots(1, 4)
    ax.ravel()[0].imshow(x_tr[data_idx, :].reshape((28,28)))
    for j in range(3):
        ax.ravel()[j+1].imshow(reconstruct_demean(eig[1][:, :k_val[j]], demean_xtr)[data_idx, :].reshape(28,28))
    [axi.set_axis_off() for axi in ax.ravel()]
    fig.suptitle('2 visualization with K = 32, 64,128')
    plt.show()

if __name__ == "__main__":
    main()
