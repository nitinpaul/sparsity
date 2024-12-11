import numpy as np
from scipy.stats import entropy


def calculate_entropy(binary_filename):
    """
    Calculates the entropy of a binary file.

    Args:
        binary_filename: Path to the binary file.

    Returns:
        The entropy of the binary data.
    """

    binaries_dir = './binaries/'

    # Load the binary file into a NumPy array
    binary_data = np.fromfile(binaries_dir + binary_filename + '.bin', dtype=np.uint8)

    # Calculate the frequency of '0's and '1's
    _, counts = np.unique(binary_data, return_counts=True)
    probabilities = counts / len(binary_data)

    # Calculate entropy using scipy.stats.entropy
    entropy_value = entropy(probabilities, base=2)  # Base 2 for bits

    return entropy_value
