import csv
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_sparsity_distribution(model_name, ax=None):
    """
    Calculates the sparsity distribution of the binary values in the 
    'predictions' column of the binaries CSV for model_name.

    Args:
        model_name: The name of the binaries file.
        ax: The axes object to plot on. If None, a new figure and axes will be created.
    """

    figures_dir = './figures/'
    binaries_dir = './binaries/'
    binaries_file = binaries_dir + model_name + '.csv'

    # Colors config
    cosfire_dark = '#ef9b20'
    cosfire_light = '#edbf33'
    densenet_dark = '#87bc45'
    densenet_light = '#bdcf32'
    neutral_yellow = '#ede15b'

    hamming_weights = []

    with open(binaries_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            predictions = eval(row['predictions'])
            hamming_weight = sum(predictions)  # Calculate Hamming weight
            hamming_weights.append(hamming_weight)

    sns.set_style("whitegrid")

    # Create the histogram using Seaborn
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))  # Create new figure and axes if not provided

    sns.histplot(
        hamming_weights,
        bins=range(min(hamming_weights), max(hamming_weights) + 2),
        ax=ax,                      
        color=cosfire_light,       # Set a consistent color
        alpha=0.8,                 # Add some transparency
        edgecolor='white',         # Add white edges to the bars
        linewidth=1,
        kde=True,                  # Add a Kernel Density Estimate for a smoother visualization
        line_kws={'linewidth': 2}  # Adjust line width of the KDE
    )

    ax.set_xlim(0, 45)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Hamming Weight', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Hamming Weight Distribution - COSFIRE (48-bit)', fontsize=14)  # Include model name in title

    # If a new figure was created, save it
    # if ax is None:

    plt.tight_layout()
    plt.savefig(figures_dir + model_name + '_hamming_weight_distribution.png', dpi=300)
    plt.close()
    