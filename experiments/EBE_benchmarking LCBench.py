import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from epoch_based_evolution import SearchSpace, Generation
import load_data

import time
import pandas as pd
import json
import os

# Function to read dataset IDs from JSON file
def read_ids_from_json(filename="./experiments/dataset_ids.json"):
    with open(filename, 'r') as file:
        dataset_ids = json.load(file)
    print(f"Loaded {len(dataset_ids)} dataset IDs from {filename}")
    return dataset_ids

# Read the list back
dataset_ids = read_ids_from_json()
# dataset_ids = [54] # for testing 
dataset_ids = [
    # 41138, 
    # 4135, # killed the GPU
    # 40981, 40996, 
    # 1111, # killed the GPU
    # 41150, 1590, 1169, 
    # 41147,  # killed the GPU
    # 1461, 1464, 40975, 41142, 1468, 40668, 
    # 1596,
    # 31, 41167, 41164, 
    # 41169, 23512, 41168, 41143, 41027, 1067, 3, 12, 1486, 23517, 1489, 40984, 40685, 41146, 54, 41166
    ]

# Define the CSV filename template
csv_filename_template = "experiments/lc_bench_results/EBE-NAS_LCBench_results_{}.csv"

# Loop through each dataset
for i, data_id in enumerate(dataset_ids):
    print(f'\nTesting for Dataset {data_id} | {i+1}/{len(dataset_ids)}')
    X_train, y_train, X_val, y_val, X_test, y_test = load_data.get_preprocessed_data(
        dataset_id=data_id, scaling=True, random_seed=13, return_as='tensor')
    input_size, output_size = load_data.get_tensor_sizes(X_train, y_train)

    search_space = SearchSpace(input_size=input_size, output_size=output_size)

    N_INDIVIDUALS = 500
    N_EPOCHS = 5
    percentile_drop = 25

    start_time = time.time()
    generation = Generation(search_space, N_INDIVIDUALS)

    all_results = []  

    for n_epoch in range(N_EPOCHS + 1):
        epoch_start_time = time.time()
        print(f'\n-Epoch: {n_epoch}')
        final_gen = generation.run_generation(X_train, y_train, X_val, y_val, percentile_drop=percentile_drop)
        num_models = len(final_gen)
        epoch_time = time.time() - epoch_start_time

        results_df = generation.return_df()
        results_df['dataset_id'] = data_id
        results_df['epoch'] = n_epoch
        results_df['epoch_time'] = epoch_time

        all_results.append(results_df)  

        if num_models <= 1:
            print(f"No models left to evaluate at epoch {n_epoch}. Stopping early.")
            break

        print(f"Survivor models: {num_models}")
        percentile_drop += 5

    final_time = time.time() - start_time
    log_message = f"Dataset {data_id} completed in {final_time:.2f} seconds\n"

    with open("process_log.txt", "a") as log_file:
        log_file.write(log_message)

    # Concatenate all results for the current dataset and save to a CSV
    if all_results:
        # Create a DataFrame from the list of results
        dataset_results_df = pd.concat(all_results, ignore_index=True)
        
        # Define the CSV filename for the current dataset
        csv_filename = csv_filename_template.format(data_id)
        
        # Save the DataFrame to a CSV file
        dataset_results_df.to_csv(csv_filename, index=False)

print("All results saved to their respective CSV files.")



