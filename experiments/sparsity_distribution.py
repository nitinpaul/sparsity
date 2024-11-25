
import csv
import matplotlib.pyplot as plt


def visualize_sparsity_distribution(model_name):
    """
    Calculates the sparsity distribution of the binary values in the 
    'predictions' column of the binaries CSV for model_name.

    Args:
        model_name: The name of the binaries file.
    """

    figures_dir = './figures/'
    binaries_dir = './binaries/'
    binaries_file = binaries_dir + model_name + '.csv'

    hamming_weights = []

    with open(binaries_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            predictions = eval(row['predictions'])
            hamming_weight = sum(predictions)  # Calculate Hamming weight
            hamming_weights.append(hamming_weight)

    # Create the histogram
    plt.hist(
        hamming_weights,
        bins=range(min(hamming_weights), max(hamming_weights) + 2),
        align='left',
        rwidth=0.8
    )
    plt.xlabel('Hamming Weight')
    plt.ylabel('Frequency')
    plt.title('Hamming Weight Distribution')
    plt.savefig(figures_dir + model_name + '_hamming_weight_distribution.png')
    