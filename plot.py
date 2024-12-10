import matplotlib.pyplot as plt
import seaborn as sns


figures_dir = './figures/'

# Colors config
cosfire_dark = '#edbf33'
cosfire_light = '#ef9b20'
densenet_dark = '#87bc45'
densenet_light = '#bdcf32'
neutral_yellow = '#ede15b'

# mAP vs Accuracy plot

bitsizes = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
cosfire_map_values = [42.22, 42.19, 86.25, 88.93, 88.92, 90.1, 88.84, 90.15, 88.69, 89.55]
densenet_map_values = [24.3, 24.2, 87.9, 87.18, 87.6, 88.92, 89.48, 89.12, 86.96, 88.86]

# Set Seaborn style for a visually appealing plot
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.plot(bitsizes, cosfire_map_values, marker='o', linestyle='-', color=cosfire_dark, label='COSFIRE')
plt.plot(bitsizes, densenet_map_values, marker='s', linestyle='-', color=densenet_dark, label='DenseNet')

# Add labels and title
plt.xlabel('Code Length (bits)', fontsize=12)
plt.ylabel('Mean Average Precision (mAP)', fontsize=12)
plt.title('Impact of Code Length on Retrieval Accuracy', fontsize=14)

# Add legend
plt.legend(fontsize=12)

# Customize the plot for better readability
plt.xticks(bitsizes, fontsize=10)  # Show all bit sizes on x-axis
plt.yticks(fontsize=10)
plt.grid(True, alpha=0.5)  # Subtle grid

# Add annotations for the points of interest (optional)
plt.annotate('COSFIRE (48 bits)', (48, 90.1), textcoords="offset points", xytext=(10,10), ha='center', fontsize=10)
plt.annotate('DenseNet (56 bits)', (56, 89.48), textcoords="offset points", xytext=(10,-20), ha='center', fontsize=10)

# Set y-axis limit to 100
plt.ylim(0, 100)

# Show the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.savefig(figures_dir + 'maps_bitsize_plot.png')
plt.close()

# Hamming Weight plot

# Hamming weights for different code lengths
hamming_weights = {
    'COSFIRE': {'48-bit': 24.04, '72-bit': 26.52},
    'DenseNet': {'56-bit': 27.22, '72-bit': 34.83}
}

# Set Seaborn style
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(6, 5))

# Bar width
bar_width = 0.15

# X positions for the bars
x_pos = [0, 1]  # One position for each method

# Plot bars for 72-bit code length
cosfire_72bit_weight = hamming_weights['COSFIRE']['72-bit']
densenet_72bit_weight = hamming_weights['DenseNet']['72-bit']
plt.bar(x_pos, [cosfire_72bit_weight, densenet_72bit_weight], width=bar_width, label='72-bit Models', color='#ede15b')
plt.annotate(str(cosfire_72bit_weight), (x_pos[0], cosfire_72bit_weight), ha='center', va='bottom', fontsize=10)
plt.annotate(str(densenet_72bit_weight), (x_pos[1], densenet_72bit_weight), ha='center', va='bottom', fontsize=10)

# Plot bar for COSFIRE 48-bit
cosfire_48bit_weight = hamming_weights['COSFIRE']['48-bit']
plt.bar(x_pos[0] - bar_width, cosfire_48bit_weight, width=bar_width, label='48-bit (COSFIRE)', color='#ef9b20')
plt.annotate(str(cosfire_48bit_weight), (x_pos[0] - bar_width, cosfire_48bit_weight), ha='center', va='bottom', fontsize=10)

# Plot bar for DenseNet 56-bit
densenet_56bit_weight = hamming_weights['DenseNet']['56-bit']
plt.bar(x_pos[1] + bar_width, densenet_56bit_weight, width=bar_width, label='56-bit (DenseNet)', color='#87bc45')
plt.annotate(str(densenet_56bit_weight), (x_pos[1] + bar_width, densenet_56bit_weight), ha='center', va='bottom', fontsize=10)

# Add labels and title
plt.xlabel('Deep Hashing Method', fontsize=12)
plt.ylabel('Average Hamming Weight', fontsize=12)
plt.title('Comparison of Average Hamming Weights', fontsize=14)

# Set x ticks and labels
plt.xticks(x_pos, ['COSFIRE', 'DenseNet'], fontsize=12)

# Add legend at the bottom center
plt.legend(fontsize=12, loc='lower center', ncol=1)  # ncol=2 arranges legend items in 2 columns

# Switch off the grid
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.savefig(figures_dir + 'hamming_weight_plot.png', dpi=600)
plt.close()

# Sparsity Ratio

# Sparsity ratios for different code lengths
sparsity_ratios = {
    'COSFIRE': {'48-bit': 0.5, '72-bit': 0.63},
    'DenseNet': {'56-bit': 0.51, '72-bit': 0.52}
}

# Set Seaborn style
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(6, 5))

# Bar width
bar_width = 0.15

# X positions for the bars
x_pos = [0, 1]  # One position for each method

# Plot bars for 72-bit code length
cosfire_72bit_weight = sparsity_ratios['COSFIRE']['72-bit']
densenet_72bit_weight = sparsity_ratios['DenseNet']['72-bit']
plt.bar(x_pos, [cosfire_72bit_weight, densenet_72bit_weight], width=bar_width, label='72-bit Models', color='#ede15b')
plt.annotate(str(cosfire_72bit_weight), (x_pos[0], cosfire_72bit_weight), ha='center', va='bottom', fontsize=10)
plt.annotate(str(densenet_72bit_weight), (x_pos[1], densenet_72bit_weight), ha='center', va='bottom', fontsize=10)

# Plot bar for COSFIRE 48-bit
cosfire_48bit_weight = sparsity_ratios['COSFIRE']['48-bit']
plt.bar(x_pos[0] - bar_width, cosfire_48bit_weight, width=bar_width, label='48-bit (COSFIRE)', color='#ef9b20')
plt.annotate(str(cosfire_48bit_weight), (x_pos[0] - bar_width, cosfire_48bit_weight), ha='center', va='bottom', fontsize=10)

# Plot bar for DenseNet 56-bit
densenet_56bit_weight = sparsity_ratios['DenseNet']['56-bit']
plt.bar(x_pos[1] + bar_width, densenet_56bit_weight, width=bar_width, label='56-bit (DenseNet)', color='#87bc45')
plt.annotate(str(densenet_56bit_weight), (x_pos[1] + bar_width, densenet_56bit_weight), ha='center', va='bottom', fontsize=10)

# Add labels and title
plt.xlabel('Deep Hashing Method', fontsize=12)
plt.ylabel('Average Sparsity Ratio', fontsize=12)
plt.title('Comparison of Average Sparsity Ratios', fontsize=14)

# Set x ticks and labels
plt.xticks(x_pos, ['COSFIRE', 'DenseNet'], fontsize=12)

# Add legend at the bottom center
plt.legend(fontsize=12, loc='lower center', ncol=1)

# Switch off the grid
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.savefig(figures_dir + 'sparsity_ratio_plot.png', dpi=600)
plt.close()
