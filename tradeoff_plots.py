import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Configuration ---
figures_dir = './figures/'

cosfire_color_base = '#ef9b20' # General COSFIRE orange
densenet_color_base = '#87bc45' # General DenseNet green

# Specific colors for each of the 4 configurations
# cosfire_72_color = '#edbf33' # Lighter orange/yellow for C72
# cosfire_48_color = '#ef9b20' # Darker orange for C48
# densenet_72_color = '#87bc45' # Darker green for D72
# densenet_56_color = '#bdcf32' # Lighter green/yellow for D56

# Define a clear mapping for the 4 points:
point_styles = {
    'C48': {'color': '#ef9b20', 'marker': 'o', 'label': 'COSFIRE 48-bit'},
    'C72': {'color': '#edbf33', 'marker': 'o', 'label': 'COSFIRE 72-bit'},
    'D56': {'color': '#bdcf32', 'marker': 's', 'label': 'DenseNet 56-bit'},
    'D72': {'color': '#87bc45', 'marker': 's', 'label': 'DenseNet 72-bit'}
}
model_short_names = ['C48', 'C72', 'D56', 'D72']


# --- Data for the four main configurations ---
# Source: Thesis report (Tables 2, 3, Figures 3, 4)
plot_data = {
    'Model_Short': model_short_names,
    'Model_Family': ['COSFIRE', 'COSFIRE', 'DenseNet', 'DenseNet'],
    'mAP (%)': [90.1, 88.69, 89.48, 86.96],
    'Code Length (bits)': [48, 72, 56, 72],
    'Avg. Hamming Wt.': [24.04, 26.52, 27.22, 34.83],
    'Avg. Sparsity Ratio': [0.50, 0.63, 0.51, 0.52],
    'Avg. Hashing Time (s)': [0.00003053, 0.00003024, 0.00104862, 0.00106079]
}
df_plot = pd.DataFrame(plot_data)

# Set Seaborn style for visually appealing plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100 # Good default DPI for saved figures
plt.rcParams['savefig.dpi'] = 300 # Higher DPI for saved figures (publication quality)

# --- Plotting Functions for Accuracy vs. Sparsity ---

def plot_map_vs_code_length(df):
    """Generates and saves a scatter plot of mAP vs. Code Length."""
    plt.figure(figsize=(8, 6))
    
    for model_short_name in model_short_names:
        row = df[df['Model_Short'] == model_short_name].iloc[0]
        style = point_styles[model_short_name]
        plt.scatter(row['Code Length (bits)'], row['mAP (%)'], 
                    color=style['color'], marker=style['marker'], s=100, label=style['label'], zorder=3)
        # Annotate points
        plt.annotate(model_short_name, 
                     (row['Code Length (bits)'], row['mAP (%)']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.xlabel('Code Length (bits)', fontsize=12)
    plt.ylabel('Mean Average Precision (mAP %)', fontsize=12)
    plt.title('Accuracy vs. Code Length', fontsize=14)
    
    min_len = df['Code Length (bits)'].min()
    max_len = df['Code Length (bits)'].max()
    plt.xlim(min_len - 8, max_len + 8)
    plt.ylim(min(df['mAP (%)']) - 1, max(df['mAP (%)']) + 1) # Dynamic Y-limits
    
    plt.legend(title='Model Configuration', fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    filename = figures_dir + 'plot_map_vs_code_length.png'
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


def plot_map_vs_avg_hamming_weight(df):
    """Generates and saves a scatter plot of mAP vs. Average Hamming Weight."""
    plt.figure(figsize=(8, 6))

    for model_short_name in model_short_names:
        row = df[df['Model_Short'] == model_short_name].iloc[0]
        style = point_styles[model_short_name]
        plt.scatter(row['Avg. Hamming Wt.'], row['mAP (%)'], 
                    color=style['color'], marker=style['marker'], s=100, label=style['label'], zorder=3)
        # Annotate points
        plt.annotate(model_short_name, 
                     (row['Avg. Hamming Wt.'], row['mAP (%)']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                     
    plt.xlabel('Average Hamming Weight', fontsize=12)
    plt.ylabel('Mean Average Precision (mAP %)', fontsize=12)
    plt.title('Accuracy vs. Average Hamming Weight', fontsize=14)

    min_hw = df['Avg. Hamming Wt.'].min()
    max_hw = df['Avg. Hamming Wt.'].max()
    plt.xlim(min_hw - 2, max_hw + 2) # Dynamic X-limits
    plt.ylim(min(df['mAP (%)']) - 1, max(df['mAP (%)']) + 1) # Dynamic Y-limits

    plt.legend(title='Model Configuration', fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    filename = figures_dir + 'plot_map_vs_avg_hamming_weight.png'
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


def plot_map_vs_avg_sparsity_ratio(df):
    """Generates and saves a scatter plot of mAP vs. Average Sparsity Ratio."""
    plt.figure(figsize=(8, 6))

    for model_short_name in model_short_names:
        row = df[df['Model_Short'] == model_short_name].iloc[0]
        style = point_styles[model_short_name]
        plt.scatter(row['Avg. Sparsity Ratio'], row['mAP (%)'], 
                    color=style['color'], marker=style['marker'], s=100, label=style['label'], zorder=3)
        # Annotate points
        plt.annotate(model_short_name, 
                     (row['Avg. Sparsity Ratio'], row['mAP (%)']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.xlabel('Average Sparsity Ratio (Proportion of Zeros)', fontsize=12)
    plt.ylabel('Mean Average Precision (mAP %)', fontsize=12)
    plt.title('Accuracy vs. Average Sparsity Ratio', fontsize=14)
    
    min_sr = df['Avg. Sparsity Ratio'].min()
    max_sr = df['Avg. Sparsity Ratio'].max()
    plt.xlim(min_sr - 0.05, max_sr + 0.05) # Dynamic X-limits
    plt.ylim(min(df['mAP (%)']) - 1, max(df['mAP (%)']) + 1) # Dynamic Y-limits

    plt.legend(title='Model Configuration', fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    filename = figures_dir + 'plot_map_vs_avg_sparsity_ratio.png'
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


# --- Plotting Functions for Efficiency vs. Sparsity ---

def plot_time_vs_code_length(df):
    """Generates and saves a scatter plot of Hashing Time vs. Code Length."""
    plt.figure(figsize=(8, 6))
    
    for model_short_name in model_short_names:
        row = df[df['Model_Short'] == model_short_name].iloc[0]
        style = point_styles[model_short_name]
        plt.scatter(row['Code Length (bits)'], row['Avg. Hashing Time (s)'], 
                    color=style['color'], marker=style['marker'], s=100, label=style['label'], zorder=3)
        # Annotate points
        plt.annotate(model_short_name, 
                     (row['Code Length (bits)'], row['Avg. Hashing Time (s)']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.xlabel('Code Length (bits)', fontsize=12)
    plt.ylabel('Average Hashing Time (s)', fontsize=12)
    plt.title('Efficiency vs. Code Length', fontsize=14)
    
    min_len = df['Code Length (bits)'].min()
    max_len = df['Code Length (bits)'].max()
    plt.xlim(min_len - 8, max_len + 8)
    # Y-axis might need scientific notation or careful formatting due to small values
    # Forcing scientific notation for y-axis if values are very small
    if df['Avg. Hashing Time (s)'].max() < 0.01:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.legend(title='Model Configuration', fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    filename = os.path.join(figures_dir, 'plot_time_vs_code_length.png')
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


def plot_time_vs_avg_hamming_weight(df):
    """Generates and saves a scatter plot of Hashing Time vs. Average Hamming Weight."""
    plt.figure(figsize=(8, 6))

    for model_short_name in model_short_names:
        row = df[df['Model_Short'] == model_short_name].iloc[0]
        style = point_styles[model_short_name]
        plt.scatter(row['Avg. Hamming Wt.'], row['Avg. Hashing Time (s)'], 
                    color=style['color'], marker=style['marker'], s=100, label=style['label'], zorder=3)
        # Annotate points
        plt.annotate(model_short_name, 
                     (row['Avg. Hamming Wt.'], row['Avg. Hashing Time (s)']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                     
    plt.xlabel('Average Hamming Weight', fontsize=12)
    plt.ylabel('Average Hashing Time (s)', fontsize=12)
    plt.title('Efficiency vs. Average Hamming Weight', fontsize=14)

    min_hw = df['Avg. Hamming Wt.'].min()
    max_hw = df['Avg. Hamming Wt.'].max()
    plt.xlim(min_hw - 2, max_hw + 2)
    if df['Avg. Hashing Time (s)'].max() < 0.01:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.legend(title='Model Configuration', fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    filename = os.path.join(figures_dir, 'plot_time_vs_avg_hamming_weight.png')
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


def plot_time_vs_avg_sparsity_ratio(df):
    """Generates and saves a scatter plot of Hashing Time vs. Average Sparsity Ratio."""
    plt.figure(figsize=(8, 6))

    for model_short_name in model_short_names:
        row = df[df['Model_Short'] == model_short_name].iloc[0]
        style = point_styles[model_short_name]
        plt.scatter(row['Avg. Sparsity Ratio'], row['Avg. Hashing Time (s)'], 
                    color=style['color'], marker=style['marker'], s=100, label=style['label'], zorder=3)
        # Annotate points
        plt.annotate(model_short_name, 
                     (row['Avg. Sparsity Ratio'], row['Avg. Hashing Time (s)']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.xlabel('Average Sparsity Ratio (Proportion of Zeros)', fontsize=12)
    plt.ylabel('Average Hashing Time (s)', fontsize=12)
    plt.title('Efficiency vs. Average Sparsity Ratio', fontsize=14)
    
    min_sr = df['Avg. Sparsity Ratio'].min()
    max_sr = df['Avg. Sparsity Ratio'].max()
    plt.xlim(min_sr - 0.05, max_sr + 0.05)
    if df['Avg. Hashing Time (s)'].max() < 0.01:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
    plt.legend(title='Model Configuration', fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    filename = os.path.join(figures_dir, 'plot_time_vs_avg_sparsity_ratio.png')
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


# --- Generate and Save Plots ---
if __name__ == '__main__':
    # Create the figures directory if it doesn't exist
    import os
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Accuracy vs. Sparsity Plots
    plot_map_vs_code_length(df_plot)
    plot_map_vs_avg_hamming_weight(df_plot)
    plot_map_vs_avg_sparsity_ratio(df_plot)

    # Efficiency vs. Sparsity Plots
    plot_time_vs_code_length(df_plot)
    plot_time_vs_avg_hamming_weight(df_plot)
    plot_time_vs_avg_sparsity_ratio(df_plot)
    
    print(f"\nAll plots saved to '{figures_dir}' directory.")
