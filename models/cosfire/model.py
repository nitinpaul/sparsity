import torch
import torch.nn as nn


class CosfireNet(nn.Module):
    """
    Neural network class implementing a Cosfire-inspired architecture with regularization.

    Args:
        input_size (int): The size of the input features.
        bitsize (int): The desired size of the output binary representation.
        l1_reg (float): The L1 regularization strength.
        l2_reg (float): The L2 regularization strength.

    Attributes:
        l1_reg (float): The L1 regularization strength.
        l2_reg (float): The L2 regularization strength.
        hd (nn.Sequential): The sequence of layers defining the network's architecture.

    COSFIRE model definition from Steven Ndung'u
    """

    def __init__(self, input_size, l1_reg, l2_reg, bitsize):
        super(CosfireNet, self).__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.hd = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.BatchNorm1d(300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.Tanh(),
            nn.Linear(200, bitsize),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the output tensor and the regularization loss.
        """
                
        regularization_loss = 0.0
        for param in self.hd.parameters():
            regularization_loss += torch.sum(torch.abs(param)) * self.l1_reg
            regularization_loss += torch.sum(param ** 2) * self.l2_reg
        return self.hd(x), regularization_loss
    

def load_model(name, l1_reg, l2_reg, bitsize):
    """
    Loads the saved model weights with name from saved_weights directory and 
    returns the initialized model

    Args:
        name: The name of the saved path file (.pth)

    Returns:
        The CosfireNet model instance loaded with the
        specified weights 
    """

    # Initialize with hyperparameters and get model instance
    model = CosfireNet(372, l1_reg, l2_reg, bitsize)

    path = './models/cosfire/saved_weights/' + name

    # Load the saved model weights
    model.load_state_dict(torch.load(path))

    return model
