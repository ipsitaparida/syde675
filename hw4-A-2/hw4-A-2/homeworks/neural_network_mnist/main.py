# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = 1 / (np.sqrt(d))
        self.weights = def_weights(alpha, h, 2, d, k)
        self.biases = def_biases(alpha, h, 2, k)
    
    
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        y = torch.matmul(x, self.weights[0].T) + self.biases[0]
        for i in range(1, len(self.weights)):
            y = torch.matmul(relu(y), self.weights[i].T) + self.biases[i]
        return y


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = 1 / (np.sqrt(d))
        self.weights = def_weights(alpha, h0, 3, d, k)
        self.biases = def_biases(alpha, h0, 3, k)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        y = torch.matmul(x, self.weights[0].T) + self.biases[0]
        for i in range(1, len(self.weights)):
            y = torch.matmul(relu(y), self.weights[i].T) + self.biases[i]
        return y

def def_weights(alpha, n_neurons, n_layers, input_dim, output_dim):
    weights = []

    # input layer weights.
    weights.append(-2 * alpha * torch.rand(n_neurons, input_dim) + alpha)
    weights[-1].requires_grad = True
    
    # hidden layer weights.
    for _ in range(n_layers - 2):
        weights.append(-2 * alpha * torch.rand(n_neurons, n_neurons) + alpha)
        weights[-1].requires_grad = True
    
    # output layer weights.
    weights.append(-2 * alpha * torch.rand(output_dim, n_neurons) + alpha)
    weights[-1].requires_grad = True
    
    return weights
		
def def_biases(alpha, n_neurons, n_layers, output_dim):
    biases = []

    # input and hidden layer biases.
    for _ in range(n_layers - 1):
        biases.append(-2 * alpha * torch.rand(n_neurons) + alpha)
        biases[-1].requires_grad = True
    
    # output layer biases.
    biases.append(-2 * alpha * torch.rand(output_dim) + alpha)
    biases[-1].requires_grad = True

    return biases

@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    losses = []
    accuracies = []
    optimizer = Adam(model.weights + model.biases)
    for i in range(100):
        loss = 0
        accuracy = 0
        for (x_batch, y) in iter(train_loader):
            pred = model.forward(x_batch)
            y_pred = torch.argmax(pred, 1)

            accuracy += torch.sum(y == y_pred)
            loss_tmp = torch.nn.functional.cross_entropy(pred, y, size_average=False)
            
            optimizer.zero_grad()
            loss_tmp.backward()
            optimizer.step()

            loss += loss_tmp
        loss /= len(train_loader.dataset)
        accuracy = accuracy.to(dtype=torch.float) / len(train_loader.dataset)

        print(f'Epoch={i}\tTraining Loss={loss}\tAccuracy={accuracy}')

        losses.append(loss)
        accuracies.append(accuracy)

        if accuracy > 0.99:
            break
    return losses

def evaluate(model: Module, test_loader: DataLoader) -> List[float]:
    accuracy = 0
    loss = 0

    for X, y in iter(test_loader):
        pred = model.forward(X)
        y_pred = torch.argmax(pred, 1)

        accuracy += torch.sum(y == y_pred)
        loss_tmp = torch.nn.functional.cross_entropy(pred, y, size_average=False)
        loss += loss_tmp

    # Normalize the performance measures.
    loss /= len(test_loader.dataset)
    accuracy = accuracy.to(dtype=torch.float) / len(test_loader.dataset)

    return accuracy, loss

def parameters(weights, biases):
	parameters = sum([np.prod(w.shape) for w in weights]) + sum([np.prod(b.shape) for b in biases])
	return parameters

@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=128, shuffle=True)
    
    # For part A - train, train losses, test accuracy, test loss, #params
    model_a = F1(64, 784, 10)
    
    train_losses_a = train(model_a, Adam, train_loader)
    model_params_a = parameters(model_a.weights, model_a.biases)
    test_accuracy_a, test_loss_a = evaluate(model_a,test_loader)
    
    # print(train_losses_a)

    # For part b - train, train losses, test accuracy, test loss, #params
    model_b = F2(32, 32, 784, 10)
    
    train_losses_b = train(model_b, Adam, train_loader)
    model_params_b = parameters(model_b.weights, model_b.biases)
    test_accuracy_b, test_loss_b = evaluate(model_b,test_loader)
    
    # print(train_losses_b)

    epochs_a = range(len(train_losses_a))
    epochs_b = range(len(train_losses_b))

    print(f'Model A test results:\nAccuracy={test_accuracy_a}\tLoss={test_loss_a}\n')
    print(f'Model B test results:\nAccuracy={test_accuracy_b}\tLoss={test_loss_b}\n')
    print(f'Model A training results:\nParameters={model_params_a}\n')
    print(f'Model B training results:\n Parameters={model_params_b}\n')

    
    plt.plot(epochs_a, [float(loss) for loss in train_losses_a], "-ko", label="Model A")
    plt.plot(epochs_b, [float(loss) for loss in train_losses_b], "-bo", label="Model B")
    plt.xlabel("epcoh")
    plt.ylabel("error")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
