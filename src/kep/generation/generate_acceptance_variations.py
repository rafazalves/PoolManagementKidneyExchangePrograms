import os

import numpy as np
import pandas as pd

from kep.constants import *

INPUT_DIR = f"./data/{SIMULATION_YEARS}_year_simulation/100_percentage/"
OUTPUT_BASE_DIR = f"./data/{SIMULATION_YEARS}_year_simulation/"
PERCENTAGES = [0, 0.25, 0.50, 0.75]
NUM_FILES =  NUMBER_OF_INSTANCES # Number of files to process (pool_1 to pool_100)

def setup_directories():
    """Create the necessary output directories if they don't exist."""
    for pct in PERCENTAGES:
        folder_name = f"{int(pct * 100)}_percentage"
        path = os.path.join(OUTPUT_BASE_DIR, folder_name)
        os.makedirs(path, exist_ok=True)

def process_pools():
    setup_directories()

    print(f"\nStarting processing of {NUM_FILES} files...")

    # Loop from pool_1 to pool_100
    for i in range(1, NUM_FILES + 1):
        filename = f"pool_{i}.csv"
        file_path = os.path.join(INPUT_DIR, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found. Skipping...")
            continue

        # Read the original CSV once per file
        try:
            df_original = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Half-Compatible rows (Compatibility == 2)
        mask = df_original['COL_PAIR_COMPATIBILITY'] == 2

        # We need the number of rows to generate random values for
        n_targets = mask.sum()
        
        # Set seed for reproducibility per file
        np.random.seed(42 + i)

        # Generate the 4 variations for this specific file
        for pct in PERCENTAGES:
            df_var = df_original.copy()

            if n_targets > 0:
                if pct == 0:
                    # If probability is 0%, they never accept. Set to -1
                    df_var.loc[mask, 'COL_PATIENT_ACCEPT_IMSUP'] = -1
                else:
                    waiting_times = np.random.geometric(p=pct, size=n_targets) - 1
                    df_var.loc[mask, 'COL_PATIENT_ACCEPT_IMSUP'] = waiting_times

            # Save to the folder
            folder_name = f"{int(pct * 100)}_percentage"
            output_path = os.path.join(OUTPUT_BASE_DIR, folder_name, filename)

            df_var.to_csv(output_path, index=False)

    print("\nAll files processed successfully.")
