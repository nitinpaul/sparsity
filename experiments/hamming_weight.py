
import csv


def calculate_average_hamming_weight(model_name):
    """
    Calculates the average Hamming weight of binary values in the 'predictions' column of a CSV file.

    Args:
        model_name: The name of the binaries file.

    Returns:
        The average Hamming weight.
    """

    binaries_dir = './binaries/'
    binaries_file = binaries_dir + model_name + '.csv'

    total_hamming_weight = 0
    num_predictions = 0

    with open(binaries_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            predictions = eval(row['predictions'])  # Safely evaluate the string as a list
            hamming_weight = sum(predictions)
            total_hamming_weight += hamming_weight
            num_predictions += 1

    average_hamming_weight = total_hamming_weight / num_predictions if num_predictions else 0
    return round(average_hamming_weight, 2)  # Round to 2 decimal places
