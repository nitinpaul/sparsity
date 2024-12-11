import csv
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat


def get_data(path, dic_labels):
       
   # Load the MATLAB file
   data = loadmat(path)
   df0 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][2])
   df2['label'] = 'Bent'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][3])
   df3['label'] = 'Compact'
   df_train = pd.concat([df0, df1, df2, df3], ignore_index=True)

   df0 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][2])
   df2['label'] = 'Bent'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][3])
   df3['label'] = 'Compact'
   df_test = pd.concat([df0, df1, df2, df3], ignore_index=True)
    
   # Rename the columns:
   column_names = ["descrip_" + str(i) for i in range(1, 401)] + ["label_code"]
   df_train.columns = column_names
   df_test.columns = column_names

   #select the optimal number of columns from the classification paper.#Get the optimal 372 descriptors only
   column_list = [f'descrip_{i}' for i in range(1, 373)] + ['label_code']
   df_train = df_train[column_list]
   df_test = df_test[column_list]

   df_train['label_code'] = df_train['label_code'].map(dic_labels)
   df_test['label_code'] = df_test['label_code'].map(dic_labels)

   return df_train, df_test


def get_verified_data(data_path, data_path_valid, data_path_test, dic_labels):
   
    train_df, valid_test_df = get_data(data_path,dic_labels)
    _, valid_prev = get_data(data_path_valid,dic_labels)
    _, test_prev = get_data(data_path_test,dic_labels)
    
    cols = list(train_df.columns[:10])
    valid_test_df['id'] = range(valid_test_df.shape[0])
    test_df = pd.merge(test_prev[cols], valid_test_df, on=cols)

    diff_set = set(np.array(valid_test_df.id)) - set(np.array(test_df.id))
    valid_df = valid_test_df[valid_test_df['id'].isin(diff_set)].copy()
    valid_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)

    # Rename label_name column:   
    train_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
    valid_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
    test_df.rename(columns = {'label_name': 'label_code'}, inplace = True)

    return train_df, valid_df, test_df


def binarize_cosfire_predictions(model_name, threshold):
    """
    Binarizes cosfire predictions and saves the results.

    Reads a CSV file containing predictions and labels, binarizes the
    predictions based on a given threshold, and saves the binarized
    predictions along with the labels to a new CSV file.

    Args:
        model_name (str): The name of the model, used for file naming.
        threshold (float): The threshold value for binarization.
            Values greater than or equal to the threshold are set to 1,
            while values less than the threshold are set to 0.
    """

    predictions_dir = './predictions/'
    binaries_dir = './binaries/'

    predictions_file = predictions_dir + model_name + '.csv'
    binaries_file = binaries_dir + model_name + '.csv'

    print(f"Reading predictions from: {predictions_file}")

    with open(predictions_file, 'r') as predictions, open(
            binaries_file, 'w', newline='') as binaries:
        reader = csv.reader(predictions)
        writer = csv.writer(binaries)

        # Write header to the output file
        header = next(reader)
        writer.writerow(header)

        num_rows_processed = 0

        for row in reader:
            predictions_str = row[0]

            # Remove spaces after '['
            predictions_str = re.sub(r"\[\s+", "[", predictions_str)
            # Remove spaces before ']'
            predictions_str = re.sub(r"\s+\]", "]", predictions_str)
            # Replace one or more spaces with a single comma
            predictions_str = re.sub(r"\s+", ",", predictions_str)

            predictions_list = [
                float(val) for val in predictions_str[1:-1].split(',')
            ]
            binarized_predictions = [
                1 if val >= threshold else 0 for val in predictions_list
            ]
            writer.writerow([binarized_predictions, row[1]])

            num_rows_processed += 1

    print(f"Binarized {num_rows_processed} rows.")
    print(f"Binarized predictions saved to: {binaries_file}")


def binarize_densenet_predictions(model_name, threshold):
    """
    Binarizes cosfire predictions and saves the results.

    Reads a CSV file containing predictions and labels, binarizes the
    predictions based on a given threshold, and saves the binarized
    predictions along with the labels to a new CSV file.

    Args:
        model_name (str): The name of the model, used for file naming.
        threshold (float): The threshold value for binarization.
            Values greater than or equal to the threshold are set to 1,
            while values less than the threshold are set to 0.
    """

    predictions_dir = './predictions/'
    binaries_dir = './binaries/'

    predictions_file = predictions_dir + model_name + '.csv'
    binaries_file = binaries_dir + model_name + '.csv'

    print(f"Reading predictions from: {predictions_file}")

    with open(predictions_file, 'r') as predictions, open(
            binaries_file, 'w', newline='') as binaries:
        reader = csv.reader(predictions)
        writer = csv.writer(binaries)

        # Write header to the output file
        header = next(reader)
        header[2] = 'label_name'  # Correct the column name
        writer.writerow(['predictions', 'label_name'])

        num_rows_processed = 0

        for row in reader:
            predictions_str = row[0]

            # Convert the string representation of the list to a numpy array
            predictions_arr = np.array(eval(predictions_str))

            # Binarize the predictions
            binarized_predictions = [
                1 if val >= threshold else 0 for val in predictions_arr
            ]

            # Write the binarized predictions and label name to the output file
            writer.writerow([binarized_predictions, row[2]]) 

            num_rows_processed += 1

    print(f"Binarized {num_rows_processed} rows.")
    print(f"Binarized predictions saved to: {binaries_file}")


def generate_binary_file(csv_filename):
    """
    Converts binary code predictions from a CSV file to a pure binary format.

    Args:
        csv_file: Path to the input CSV file.
    """

    binaries_dir = './binaries/'

    # Read the CSV file using pandas, only loading the 'predictions' column
    df = pd.read_csv(binaries_dir + csv_filename + '.csv', usecols=['predictions'])

    # Convert the string representation of binary codes to NumPy arrays
    binary_codes = df['predictions'].apply(lambda x: np.fromstring(x[1:-1], dtype=np.uint8, sep=', '))

    # Stack the arrays into a single NumPy array
    binary_codes = np.vstack(binary_codes)

    # Save the NumPy array as a binary file
    binary_codes.tofile(binaries_dir + csv_filename + '.bin')


