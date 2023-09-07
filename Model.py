################################################################################
#                                                                              #
# This code defines two PyTorch neural network models. The LinearRegression    #
# model predicts a single output from a single input using a linear layer. The #
# ANNModel (Artificial Neural Network) model can have multiple hidden layers,  #
# with the option to customize the number and size of these layers. The models #
# are constructed using PyTorch's neural network module and include various    #
# linear and activation layers for forward computations.                       #
#                                                                              #
################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class LinearRegression(nn.Module):
    """
        A linear regression model implemented as a neural network module.

    Args:
        nn (module): The PyTorch nn module.
    """

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Performs a forward pass through the model. since we are only
            using a Linear layer this is a linear regression model.
            NOTE: This does not need a activation function
        Args:
            x (tensor): Input tensor. Shape (Batch_size, 1) By default the batch size is 10

        Returns:
            tensor: output tensor.
        """
        x = self.fc1(x)
        return x


class ANNModel(nn.Module):
    """
    An Artificial Neural Network (ANN) model implemented as a neural network module.

    Args:
        nn (module): The PyTorch nn module.
    """

    def __init__(
        self,
        n_hidden: int = 2,
        hidden_size: int = 15,
        activation: nn.Module = nn.Sigmoid,
    ):
        """
            Initializes the ANNModel. with n_hidden hidden layers and hidden_size neurons in
            each layer activation is the activation function applied after each layer to increase
            model's capacity

        Args:
            n_hidden (int, optional): Number of hidden layers. Defaults to 2.
            hidden_size (int, optional): Size of hidden layers. Defaults to 15.
            activation (module, optional): Activation function module. Defaults to nn.Sigmoid.
        """
        super().__init__()
        self.input = nn.Linear(1, hidden_size)

        modules = [nn.Linear(hidden_size, hidden_size), activation] * n_hidden

        self.hidden = nn.Sequential(*modules)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Performs a forward pass through the model.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Model's output tensor.
        """
        x = self.input(x)
        x = self.hidden(x)
        x = self.out(x)
        return x
