import csv


def calculate_average_sparsity_ratio(model_name):
    """
    Calculates the sparsity ratio (number of 0s / total code length) of binary
    values in the 'predictions' column of the binaries CSV for model_name.

    Args:
        model_name: The name of the binaries file.

    Returns:
        The average sparsity ratio.
    """

    binaries_dir = './binaries/'
    binaries_file = binaries_dir + model_name + '.csv'

    total_sparsity_ratio = 0
    num_rows = 0

    with open(binaries_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            predictions = eval(row['predictions'])

            # Calculate sparsity ratio for the current row
            num_zeros = predictions.count(0)
            code_length = len(predictions)
            sparsity_ratio = num_zeros / code_length
            total_sparsity_ratio += sparsity_ratio
            num_rows += 1

    # Calculate average sparsity ratio
    average_sparsity_ratio = total_sparsity_ratio / num_rows if num_rows else 0
    return round(average_sparsity_ratio, 2) # Round to 2 decimal places
