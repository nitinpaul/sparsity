import time
import numpy as np
import torch 
from torch.utils.data import DataLoader

from .utils import get_verified_data


dic_labels = { 
    'FRI':0,
    'FRII':1,
    'Bent':2,
    'Compact':3,
}

def calculate_average_hashing_time(model, data, device='cuda'):
    """
    Runs the hashing time experiment on the given cosfire model and data.

    Args:
        model: The model to evaluate.
        data: The data module containing the datasets.
        device (str): The device to run the model on ('cuda' or 'cpu')

    Returns:
        tuple: (average_hashing_time_per_image, std_dev_hashing_time)
               average_hashing_time_per_image (float): Avg time in seconds.
               std_dev_hashing_time (float): Standard deviation across runs.
    """
    
    # Set model to evaluation mode
    model.eval()
    model.to(device) # Move model to the specified device

    # Set params
    batch_size = 32

    # Set path to descriptors
    full_data_path = './models/cosfire/descriptors/train_valid_test.mat'
    test_data_path = './models/cosfire/descriptors/train_test.mat' 
    validation_data_path = './models/cosfire/descriptors/train_valid.mat'

    _, _, test_df = get_verified_data(
        full_data_path, validation_data_path, test_data_path, dic_labels
    )

    test_dataset = data.CosfireDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_images = len(test_dataset)

    # --- Hashing Time Measurement ---

    num_runs = 7 # Number of times to repeat the experiment for averaging
    num_warmup = 3 # Number of initial runs to discard for warm-up
    run_avg_times = [] # Stores the average time per image for each timed run

    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Number of test images: {num_images}")
    print(f"Batch size: {batch_size}")
    print(f"Number of warm-up runs: {num_warmup}")
    print(f"Number of timed runs: {num_runs}")
    print("-" * 30)

    with torch.no_grad(): # Disable gradient calculations for inference

        for run in range(num_runs + num_warmup):

            batch_times = [] # Stores time taken for each batch in this run
            total_images_processed_run = 0

            # Synchronize CUDA device before starting timer for more accurate GPU timing
            if device == 'cuda':
                torch.cuda.synchronize()

            start_time_run = time.perf_counter()

            for batch_data, _ in test_dataloader:

                batch_data = batch_data.to(device)
                current_batch_size = batch_data.size(0)
                total_images_processed_run += current_batch_size

                # Synchronize CUDA device before timing the operation
                if device == 'cuda':
                    torch.cuda.synchronize()

                start_time_batch = time.perf_counter()

                # --- Hashing Operation ---
                _ = model(batch_data) # Perform the forward pass (hashing)
                # --- End Hashing Operation ---

                # Synchronize CUDA device after the operation before stopping timer
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time_batch = time.perf_counter()

                batch_times.append(end_time_batch - start_time_batch)

            # Synchronize CUDA device after finishing all batches
            if device == 'cuda':
                 torch.cuda.synchronize()
            end_time_run = time.perf_counter()

            total_run_time = end_time_run - start_time_run

            # Calculate average time per image for this run
            # Sum of all batch times / total images processed
            if total_images_processed_run > 0:
                 avg_time_per_image_run = sum(batch_times) / total_images_processed_run
            else:
                 avg_time_per_image_run = 0

            if run >= num_warmup:
                run_avg_times.append(avg_time_per_image_run)
                print(f"Run {run - num_warmup + 1}/{num_runs} | Avg Time/Image: {avg_time_per_image_run:.8f} sec | Total Run Time: {total_run_time:.4f} sec")
            else:
                print(f"Warm-up Run {run + 1}/{num_warmup} | Avg Time/Image: {avg_time_per_image_run:.8f} sec | Total Run Time: {total_run_time:.4f} sec")

    # Calculate the final average and standard deviation across the timed runs
    if run_avg_times:
        final_avg_hashing_time = np.mean(run_avg_times)
        std_dev_hashing_time = np.std(run_avg_times)
    else:
        print("Error: No timed runs were completed.")
        final_avg_hashing_time = None
        std_dev_hashing_time = None

    print("-" * 30)

    return final_avg_hashing_time, std_dev_hashing_time
