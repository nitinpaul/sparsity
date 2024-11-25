import os
import random

import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import CosfireNet
from data import CosfireDataset, get_data_split

pd.options.mode.chained_assignment = None


def dsh_loss(u, y, alpha, margin):
    """
    Calculates the Deep Supervised Hashing (DSH) loss.

    This function implements the DSH loss function as defined in the paper by Liu et al.,
    with modifications for numerical stability and efficiency.

    Args:
        u (torch.Tensor): The output of the neural network (embeddings).
        y (torch.Tensor): The ground truth similarity labels (0 for similar, 1 for dissimilar).
        alpha (float): Weighting parameter for the regularization term.
        margin (float): Margin threshold parameter (m in the paper). The margin defines a radius
                        around X1,X2. Dissimilar pairs contribute to the loss function only if 
                        their distance is within this radius.

    Returns:
        torch.Tensor: The calculated DSH loss.
    """

    # Ensure y is of type integer
    y = y.int()

    # Expand y to create a pairwise similarity matrix
    y = y.unsqueeze(1).expand(len(y), len(y))

    # Create a pairwise similarity label matrix (0 for similar, 1 for dissimilar)
    y_label = torch.ones_like(torch.empty(len(y), len(y)), device=u.device)
    y_label[y == y.t()] = 0

    # Calculate the pairwise squared Euclidean distance
    dist = torch.cdist(u, u, p=2).pow(2)

    # Calculate the margin parameter according to the paper
    m = 2 * margin

    # Calculate the three loss terms (similar to Eqn. (4) in the paper)
    loss1 = 0.5 * (1 - y_label) * dist  # Penalizes similar pairs with large distances
    loss2 = 0.5 * y_label * torch.clamp(
        m - dist, min=0
    )  # Penalizes dissimilar pairs with small distances
    B1 = torch.norm(torch.abs(u) - 1, p=1, dim=1)
    B2 = B1.unsqueeze(1).expand(len(y), len(y))
    loss3 = (B2 + B2.t()) * alpha  # Regularization term to encourage binary-like outputs

    # Calculate the mean minibatch loss
    minibatch_loss = torch.mean(loss1 + loss2 + loss3)

    return minibatch_loss


def run(learning_rate=0.01, l1_reg=1e-08, l2_reg=1e-08, bitsize=72, save_model=True, model_name='base_model'):
    """
    Trains the passed model for the specified epochs, with the specified learning rate

    Args:
        learning_rate: Learning rate to train with
        save_model: Save the model weights

    Returns:
        The trained model object
    """

    # Reproducibility settings
    seed = 100
    random.seed(seed) # Python's built-in random module
    os.environ['PYTHONHASHSEED'] = str(seed) # Environment variable for Python hash seed
    np.random.seed(seed) # NumPy random seed
    torch.manual_seed(seed) # PyTorch random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # Set seed for GPU
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
        torch.backends.cudnn.deterministic = True # Ensure deterministic algorithms
        torch.backends.cudnn.benchmark = False # Control benchmarking

    # Initialize model
    input_size = 372
    model = CosfireNet(input_size, l1_reg, l2_reg, bitsize)
    
    # Output directory
    output_dir = './output/'

    # Directory to save the trained weights
    saved_weights_dir = './saved_weights/'

    # Set params
    epochs=2000
    learning_rate=0.01
    batch_size = 32
    alpha = 0.001
    margin = 36

    # Set path to descriptors
    full_data_path = './descriptors/train_valid_test.mat'
    # test_data_path = './descriptors/train_test.mat' 
    validation_data_path = './descriptors/train_valid.mat'

    train_df, valid_test_df = get_data_split(full_data_path)
    _, test_prev = get_data_split(validation_data_path)

    cols = list(train_df.columns[:10])
    valid_test_df['id'] = range(valid_test_df.shape[0])
    test_df = pd.merge(test_prev[cols], valid_test_df, on=cols)
    diff_set = set(np.array(valid_test_df.id)) - set(np.array(test_df.id))
    valid_df = valid_test_df[valid_test_df['id'].isin(diff_set)]

    valid_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)
  
    train_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
    valid_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
    test_df.rename(columns = {'label_name': 'label_code'}, inplace = True)

    # DataLoader for training set
    train_dataset = CosfireDataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # DataLoader for validation set
    valid_dataset = CosfireDataset(valid_df)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Check if GPU is available and move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for _ in range(epochs):

        model.train()
        total_train_loss = 0.0
        for _, (train_inputs, labels) in enumerate(train_dataloader):

            # Move data to the same device as the model
            train_inputs, labels = train_inputs.to(device), labels.to(device) 

            # optimizer.zero_grad() prevents loss.backward() from accumulating the 
            # new gradient values with the ones from the previous step.
            optimizer.zero_grad()
            train_outputs, reg_loss=model(train_inputs)
            loss = dsh_loss(u=train_outputs, y=labels, alpha=alpha, margin=margin) + reg_loss
            loss.backward() # update the gradient
            optimizer.step() # update the weights
            total_train_loss += loss.item() * train_inputs.size(0)

        scheduler.step()

        # Calculate average training loss
        average_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(average_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in valid_dataloader:

                # Move validation data to the same device as the model
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs, reg_loss=model(val_inputs)
                val_loss = dsh_loss(u=val_outputs, y=val_labels, alpha=alpha, margin=margin) + reg_loss
                total_val_loss += val_loss.item() * val_inputs.size(0)

        # Calculate average validation loss
        average_val_loss = total_val_loss / len(valid_dataloader)
        val_losses.append(average_val_loss)

    # Plotting
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(output_dir + '/Train_Validation_Loss_bitsize_' + str(bitsize) + '_LR_' + str(learning_rate) + '.png')
    plt.close()

    if save_model:
        torch.save(model.state_dict(), saved_weights_dir + model_name + '.pth')

    return model

if __name__ == '__main__':
    run(l1_reg=1e-08, l2_reg=1e-08, bitsize=80, model_name='cosfire_80bit_lr0p01')
