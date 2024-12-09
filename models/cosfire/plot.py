import matplotlib.pyplot as plt
import seaborn as sns


bitsizes = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
cosfire_map_values = [42.22, 42.19, 86.25, 88.93, 88.92, 90.1, 88.84, 90.15, 88.69, 89.55]
densenet_map_values = [24.3, 24.2, 87.9, 87.18, 87.6, 88.92, 89.48, 89.12, 86.96, 88.86]

# Set Seaborn style for a visually appealing plot
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.plot(bitsizes, cosfire_map_values, marker='o', linestyle='-', color='red', label='COSFIRE')
plt.plot(bitsizes, densenet_map_values, marker='s', linestyle='-', color='blue', label='DenseNet')

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
plt.annotate('COSFIRE (48 bits)', (48, 90.1), textcoords="offset points", xytext=(10,10), ha='center', fontsize=10, arrowprops=dict(arrowstyle="->"))
plt.annotate('DenseNet (56 bits)', (56, 89.48), textcoords="offset points", xytext=(10,-20), ha='center', fontsize=10, arrowprops=dict(arrowstyle="->"))

# Set y-axis limit to 100
plt.ylim(0, 100)

# Show the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.savefig('maps_bitsize_plot.png')
plt.close()
