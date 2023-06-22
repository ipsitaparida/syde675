if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    You should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _)  , (x_test, _) = load_dataset("mnist")
    # print(x_train.shape)

    # Part b , k =10
    centers = lloyd_algorithm(x_train, 10)

    fig, ax = plt.subplots(2, 5)
    # print(centers)
    for i in range(len(centers)):
        center_img = centers[i].reshape((28, 28))
        ax.ravel()[i].imshow(center_img, cmap='gray')
        [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()    

    # Part c
    k_vals = [2, 4, 8, 16, 32, 64]

    train_errors = []
    test_errors = []

    for k in k_vals:
        centers = lloyd_algorithm(x_train, k)
        train_errors.append(calculate_error(x_train, centers))
        test_errors.append(calculate_error(x_test, centers))

    plt.plot(k_vals, train_errors, '-ro', label='Train error')
    plt.plot(k_vals, test_errors, '-gx', label='Train error')
    plt.xlabel('# clusters(k)')
    plt.ylabel("Error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
