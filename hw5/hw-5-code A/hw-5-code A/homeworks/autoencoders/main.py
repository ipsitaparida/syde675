import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def F1(h: int) -> nn.Module:
    """Model F1, it should performs an operation W_d * W_e * x as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    return nn.Sequential(
        nn.Linear(784, h),
        nn.Linear(h, 784)
    )


@problem.tag("hw4-A")
def F2(h: int) -> nn.Module:
    """Model F1, it should performs an operation ReLU(W_d * ReLU(W_e * x)) as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    return nn.Sequential(
        nn.Linear(784, h),
        nn.ReLU(),
        nn.Linear(h, 784),
        nn.ReLU()
    )


@problem.tag("hw4-A")
def train(
    model: nn.Module, optimizer: Adam, train_loader: DataLoader, epochs: int = 40
) -> float:
    """
    Train a model until convergence on train set, and return a mean squared error loss on the last epoch.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
            Hint: You can try using learning rate of 5e-5.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Note:
        - Unfortunately due to how DataLoader class is implemented in PyTorch
            "for x_batch in train_loader:" will not work. Use:
            "for (x_batch,) in train_loader:" instead.

    Returns:
        float: Final training error/loss
    """
    lossfn = nn.MSELoss()
    mseloss = float(0)
    for i in tqdm(range(epochs)):
        for (x_batch, ) in iter(train_loader):
            optimizer.zero_grad()
            pred = model(x_batch)

            if(i == epochs-1) :
                loss = lossfn(pred, x_batch)
                loss.backward()
                mseloss = loss.item()
            optimizer.step()
    return mseloss       

@problem.tag("hw4-A")
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluates a model on a provided dataset.
    It should return an average loss of that dataset.

    Args:
        model (Module): TRAINED Model to evaluate. Either F1, or F2 in this problem.
        loader (DataLoader): DataLoader with some data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Returns:
        float: Mean Squared Error on the provided dataset.
    """
    lossfn = nn.MSELoss()
    model.eval()
    test_mseloss = 0.0
    count = 0
    with torch.no_grad():
        for (x_batch,) in loader:
            pred = model(x_batch)
            loss = lossfn(pred, x_batch)
            test_mseloss += loss.item()
            count += 1
    
    test_mseloss = test_mseloss / count
    return test_mseloss


@problem.tag("hw4-A", start_line=9)
def main():
    """
    Main function of autoencoders problem.

    It should:
        A. Train an F1 model with hs 32, 64, 128, report loss of the last epoch
            and visualize reconstructions of 10 images side-by-side with original images.
        B. Same as A, but with F2 model
        C. Use models from parts A and B with h=128, and report reconstruction error (MSE) on test set.

    Note:
        - For visualizing images feel free to use images_to_visualize variable.
            It is a FloatTensor of shape (10, 784).
        - For having multiple axes on a single plot you can use plt.subplots function
        - For visualizing an image you can use plt.imshow (or ax.imshow if ax is an axis)
    """
    (x_train, y_train), (x_test, _) = load_dataset("mnist")
    x = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    # Neat little line that gives you one image per digit for visualization in parts a and b
    images_to_visualize = x[[np.argwhere(y_train == i)[0][0] for i in range(10)]]

    train_loader = DataLoader(TensorDataset(x), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=32, shuffle=True)
    
    h_arr = [32, 64, 128]
    # # Part a
    for h in h_arr:
            model = F1(h)
            optimizer = Adam(params=model.parameters(), lr=5e-5)
            e = train(model, optimizer, train_loader, 40)
            
            fig, ax = plt.subplots(10, 2)
            for i in range(10):
                ax.ravel()[i*2].imshow(images_to_visualize[i].reshape((28,28)))
                ax.ravel()[i*2+1].imshow(model(images_to_visualize[i]).detach().numpy().reshape(28, 28))
            [axi.set_axis_off() for axi in ax.ravel()]
            fig.suptitle('F1 Reconstruction with h = ' + str(h))
            plt.show()
    # Part b
    for h in h_arr:
            model = F2(h)
            optimizer = Adam(params=model.parameters(), lr=5e-5)
            train(model, optimizer, train_loader, 40)
            
            fig, ax = plt.subplots(10, 2)
            for i in range(10):
                ax.ravel()[i*2].imshow(images_to_visualize[i].reshape((28,28)))
                ax.ravel()[i*2+1].imshow(model(images_to_visualize[i]).detach().numpy().reshape(28, 28))
            [axi.set_axis_off() for axi in ax.ravel()]
            fig.suptitle('F2 Reconstruction with h = ' + str(h))
            plt.show()
    # Part c
    for m in [F1, F2]:
            model = m(128)
            optimizer = Adam(params=model.parameters(), lr=5e-5)
            train(model, optimizer, train_loader, 40)
            val_test = evaluate(model, test_loader)
            print(val_test)

    # Part d
    data_idx = 2
    fig, ax = plt.subplots(1, 4)
    j = 0
    for h in h_arr:
        model = F1(h) # run for both F1 and F2
        optimizer = Adam(params=model.parameters(), lr=5e-5)
        train(model, optimizer, train_loader, 40)

        ax.ravel()[0].imshow(images_to_visualize[data_idx].reshape((28,28)))
        ax.ravel()[j+1].imshow(model(images_to_visualize[data_idx]).detach().numpy().reshape(28, 28))
        j += 1
    [axi.set_axis_off() for axi in ax.ravel()]
    fig.suptitle('2\'s F1 Reconstruction with h = ' + str(h_arr))
    plt.show()

if __name__ == "__main__":
    main()
