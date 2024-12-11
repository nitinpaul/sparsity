import os
from  itertools import groupby

import numpy as np
from scipy.stats import entropy


binaries_dir = './binaries/'

def calculate_entropy(binary_filename):
    """
    Calculates the entropy of a binary file.

    Args:
        binary_filename: Path to the binary file.

    Returns:
        The entropy of the binary data.
    """

    # Load the binary file into a NumPy array
    binary_data = np.fromfile(binaries_dir + binary_filename + '.bin', dtype=np.uint8)

    # Calculate the frequency of '0's and '1's
    _, counts = np.unique(binary_data, return_counts=True)
    probabilities = counts / len(binary_data)

    # Calculate entropy using scipy.stats.entropy
    entropy_value = entropy(probabilities, base=2)  # Base 2 for bits

    return entropy_value


def rle_encode(data):
    """
    Encodes data using run-length encoding.

    Args:
        data: A sequence of data (e.g., a list or NumPy array).

    Returns:
        A list of tuples, where each tuple represents (value, count).
    """
    return [(k, sum(1 for _ in g)) for k, g in groupby(data)]


def rle_decode(encoded_data):
    """
    Decodes data encoded with run-length encoding.

    Args:
        encoded_data: A list of tuples, where each tuple is (value, count).

    Returns:
        The decoded data sequence.

    Usage:
        encoded_code = rle_encode(binary_code)  # [(1, 1), (0, 4), (1, 3), (0, 2)]
        decoded_code = rle_decode(encoded_code)  # [1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    """
    return [value for value, count in encoded_data for _ in range(count)]


def compress_binary_file(input_filename, compression_method):
    """
    Compresses a binary file using the specified method.

    Args:
        input_file: Input binary filename.
        compression_method: The compression method ('rle', 'huffman', or 'zlib').
    """

    # Load the binary data
    binary_data = np.fromfile(binaries_dir + input_filename + '.bin', dtype=np.uint8)

    if compression_method == 'rle':
        encoded_data = rle_encode(binary_data)
        
        # Convert encoded_data to a NumPy array of tuples
        encoded_data = np.array(encoded_data, dtype=np.dtype('uint8, uint32'))
    else:
        raise ValueError("Invalid compression method.")
    
    output_filename = input_filename + '_' + compression_method

    # Save the compressed data to a binary file
    with open(binaries_dir + output_filename + '.bin', 'wb') as f:
        f.write(encoded_data)

    # Calculate compression ratio
    compression_ratio = round(get_compression_ratio(input_filename, output_filename), 2)
    space_savings = round((1 - compression_ratio) * 100, 2)

    return compression_ratio, space_savings


def get_compression_ratio(original_filename, compressed_filename):
    """
    Calculates the compression ratio.

    Args:
        original_file: Path to the original file.
        compressed_file: Path to the compressed file.

    Returns:
        The compression ratio (original size / compressed size). 
    """
    original_size = os.path.getsize(binaries_dir + original_filename + '.bin')
    compressed_size = os.path.getsize(binaries_dir + compressed_filename + '.bin')

    return compressed_size / original_size
