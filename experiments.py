import argparse
import os
import random

import numpy as np
import torch

from models.cosfire import data as cosfire_data
from models.cosfire.model import load_model as load_cosfire_model
from models.densenet.model import load_model as load_densenet_model
from experiments.accuracy import evaluate_cosfire_accuracy
from experiments.accuracy import evaluate_densenet_accuracy
from experiments.hamming_weight import calculate_average_hamming_weight
from experiments.sparsity_ratio import calculate_average_sparsity_ratio
from experiments.sparsity_distribution import visualize_sparsity_distribution
from experiments.utils import binarize_cosfire_predictions
from experiments.utils import binarize_densenet_predictions


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

# Predictions directory
predictions_dir = './predictions/'


# -------------------------------------
# EXPERIMENT 1: Accuracy
# -------------------------------------

def accuracy(model_name):
    """
    Calculates and prints the mean average precision for the specified model.

    This function takes the model name and the name of the model weights file name
    as input, loads the model, and calculates the mean average precision (mAP) on
    the test set. The calculated mAP is then printed to the console.

    Args:
        model_name (str): The name of the model type.
        model_weights (str): The name of the model weights file.
    """

    # Get params from model name string
    slices = model_name.split('_')
    model_type = slices[0]
    bitsize = int(slices[1].replace('bit', ''))
    model_weights_filename = model_name + '.pth'

    if model_type == 'cosfire':

        model = load_cosfire_model(
            name=model_weights_filename, bitsize=bitsize, l1_reg=1e-08, l2_reg=1e-08
        )
        mAP, threshold_max_mAP, predictions = evaluate_cosfire_accuracy(
            model, cosfire_data
        )

        print("---------------------------------------")
        print("Model name: " + model_name)
        print("mAP:", mAP)
        print("Threshold:", threshold_max_mAP)
        print("---------------------------------------")

        # Write predictions to file
        predictions.to_csv(predictions_dir + model_name + '.csv', index=False)

    elif model_type == 'densenet':
    
        model = load_densenet_model(model_weights_filename, bitsize)
        mAP = evaluate_densenet_accuracy(model)


# -------------------------------------
# EXPERIMENT 3: Hamming Weight 
# -------------------------------------

def hamming_weight(model_name):

    average_hamming_weight = calculate_average_hamming_weight(model_name)

    print("---------------------------------------")
    print("Model name: " + model_name)
    print("Average Hamming Weight:", average_hamming_weight)
    print("---------------------------------------")


# -------------------------------------
# EXPERIMENT 4: Sparsity Ratio  
# -------------------------------------

def sparsity_ratio(model_name):
    
    average_sparsity_ratio = calculate_average_sparsity_ratio(model_name)

    print("---------------------------------------")
    print("Model name: " + model_name)
    print("Sparsity Ratio:", average_sparsity_ratio)
    print("---------------------------------------")


# -------------------------------------
# EXPERIMENT 5: Sparsity Distribution 
# -------------------------------------

# def sparsity_distribution(model_name):
#     print("Correct function")
    
#     visualize_sparsity_distribution(model_name)


# CLI Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run the experiments")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for binarize function
    accuracy_parser = subparsers.add_parser("binarize")
    accuracy_parser.add_argument(
        "model_name", type=str, help="The name of the model to binarize the predictions for"
    )
    accuracy_parser.add_argument(
        "threshold", type=float, help="The threshold corresponding to the maximum mAP value"
    )

    # Subparser for the accuracy experiment
    accuracy_parser = subparsers.add_parser("accuracy")
    accuracy_parser.add_argument(
        "model_name", type=str, help="The name of the model weights pth file"
    )

    # Subparser for the code length experiment
    accuracy_parser = subparsers.add_parser("hamming-weight")
    accuracy_parser.add_argument(
        "model_name", type=str, help="The name of the model weights pth file"
    )

    # Subparser for the sparsity ratio experiment
    accuracy_parser = subparsers.add_parser("sparsity-ratio")
    accuracy_parser.add_argument(
        "model_name", type=str, help="The name of the model weights pth file"
    )

    # Subparser for the sparsity distribution experiment
    accuracy_parser = subparsers.add_parser("sparsity-distribution")
    accuracy_parser.add_argument(
        "model_name", type=str, help="The name of the model weights pth file"
    )

    args = parser.parse_args()

    if args.command == 'accuracy':
        accuracy(args.model_name)
    elif args.command == 'binarize':
        if args.model_name.split('_')[0] == 'cosfire':
            binarize_cosfire_predictions(args.model_name, args.threshold)
        else:
            binarize_densenet_predictions(args.model_name, args.threshold)
    elif args.command == 'hamming-weight':
        hamming_weight(args.model_name)
    elif args.command == 'sparsity-ratio':
        sparsity_ratio(args.model_name)
    elif args.command == 'sparsity-distribution':
        visualize_sparsity_distribution(args.model_name, None)
