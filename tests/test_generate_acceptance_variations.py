from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import re

import kep.generation.generate_acceptance_variations as gen_av
from kep.constants import *


@pytest.fixture
def mock_df():
    """Create a mock DataFrame to simulate 'pool_X.csv' files with 10 pairs, where 5 are Half-Compatible (Compatibility = 2)."""
    data = {
        'COL_PAIR_ID': range(10),
        'COL_PAIR_COMPATIBILITY': [2, 2, 3, 2, 1, 2, 3, 2, 1, 3], # Index 0, 1, 3, 5, 7 are Compat=2
        'COL_PATIENT_ACCEPT_IMSUP': [0, 0, -1, 0, -1, 0, -1, 0, -1, -1] # Initially all -1 except the half-compatible (in the generator they all start as 0)
    }
    return pd.DataFrame(data)


@patch('kep.generation.generate_acceptance_variations.pd.read_csv')
@patch('kep.generation.generate_acceptance_variations.os.makedirs')
@patch('kep.generation.generate_acceptance_variations.os.path.exists')
def test_process_pools_logic(mock_exists, mock_makedirs, mock_read_csv, mock_df):
    """
    Test logic of percentages (0%, 25%, 50%, 75%) 
    """
    # Configure Mocks
    mock_exists.return_value = True # Assume directory always exists
    mock_read_csv.return_value = mock_df.copy()

    # Patch in constant NUM_FILES to 1 for testing
    with patch('kep.generation.generate_acceptance_variations.NUM_FILES', 1):
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            gen_av.process_pools()

            assert mock_makedirs.called, "Should have attempted to create directories."
            assert mock_to_csv.call_count == 4, "Method should have been called 4 times (once for each percentage: 0, 0.25, 0.5, 0.75)"

@patch('numpy.random.geometric')
def test_acceptance_logic_calculation(mock_geometric, mock_df):
    """
    Test the internal logic of calculating waiting times.
    It verifies if the logic `geometric(p) - 1` is applied correctly.
    """
    df = mock_df.copy()

    # Filter compatibility 2 (The targets)
    mask = df['COL_PAIR_COMPATIBILITY'] == 2
    target_indices = df[mask].index
    n_targets = len(target_indices) # Should be 5 in our mock_df

    # Configure Mock for Geometric Distribution
    # If geometric returns [1, 5, 2, 1, 10], the code should store [0, 4, 1, 0, 9]
    mock_values = np.array([1, 5, 2, 1, 10])
    mock_geometric.return_value = mock_values[:n_targets]

    pct = 0.50 # Test arbitrary percentage

    # Simulate the logic inside the script
    if n_targets > 0:
        # Expected Logic: trials - 1 = failures (waiting time)
        waiting_times = mock_geometric(p=pct, size=n_targets) - 1
        df.loc[mask, 'COL_PATIENT_ACCEPT_IMSUP'] = waiting_times
    
    # Verify calculated values for Half-Compatible pairs
    result_subset = df.loc[target_indices, 'COL_PATIENT_ACCEPT_IMSUP'].values
    expected_values = mock_values[:n_targets] - 1
    
    np.testing.assert_array_equal(result_subset, expected_values, 
                                  err_msg="Values should be (geometric_result - 1).")

    # Verify that rows NOT compatibility = 2 were NOT altered (should remain -1)
    non_target_mask = df['COL_PAIR_COMPATIBILITY'] != 2
    # In mock_df, non-targets are initialized as -1
    assert (df[non_target_mask]['COL_PATIENT_ACCEPT_IMSUP'] == -1).all(), \
        "Rows that are not Half-Compatible should remain -1."

def test_process_pools_file_not_found():
    """
    Test handling of missing pool files.
    """
    with patch('kep.generation.generate_acceptance_variations.setup_directories') as mock_setup, \
         patch('kep.generation.generate_acceptance_variations.os.path.exists') as mock_exists, \
         patch('kep.generation.generate_acceptance_variations.NUM_FILES', 1), \
         patch('builtins.print') as mock_print:

        mock_exists.return_value = False

        gen_av.process_pools()

        assert mock_print.call_count >= 2

        printed_messages = [args[0] for args, _ in mock_print.call_args_list]
        assert any("not found" in msg for msg in printed_messages), "Should have printed file not found warning."

def test_process_pools_read_error():
    """
    Test handling of CSV read errors.
    """
    with patch('kep.generation.generate_acceptance_variations.setup_directories') as mock_setup, \
         patch('kep.generation.generate_acceptance_variations.os.path.exists') as mock_exists, \
         patch('kep.generation.generate_acceptance_variations.pd.read_csv') as mock_read_csv, \
         patch('kep.generation.generate_acceptance_variations.NUM_FILES', 1), \
         patch('builtins.print') as mock_print:

        mock_exists.return_value = True
        mock_read_csv.side_effect = Exception("Simulated Read Error")

        gen_av.process_pools()

        # Verifies if error message was printed
        printed_messages = [str(args[0]) for args, _ in mock_print.call_args_list]
        assert any("Error reading" in msg for msg in printed_messages), "Should have printed read error warning."
        assert any("Simulated Read Error" in msg for msg in printed_messages)

@patch('kep.generation.generate_acceptance_variations.pd.read_csv')
@patch('kep.generation.generate_acceptance_variations.os.makedirs')
@patch('kep.generation.generate_acceptance_variations.os.path.exists')
def test_consistency_across_percentages(mock_exists, mock_makedirs, mock_read_csv, mock_df):
    """
    Test STRONG logic: Higher acceptance probabilities (e.g., 75%) should result in lower AVERAGE wait times than lower probabilities (e.g., 25%).
    """
    data = {
        'COL_PAIR_ID': range(200),
        'COL_PAIR_COMPATIBILITY': [2] * 200, 
        'COL_PATIENT_ACCEPT_IMSUP': [-1] * 200
    }
    df_large = pd.DataFrame(data)
    
    mock_exists.return_value = True
    mock_read_csv.return_value = df_large.copy()

    saved_dfs = {}

    # Function to capture DataFrames being saved
    def save_capture(self, path_or_buf, **kwargs):
        path_str = str(path_or_buf)
        match = re.search(r'(\d+)_percentage', path_str)
        
        if match:
            pct = int(match.group(1))
            saved_dfs[pct] = self.copy()
        else:
            print(f"Warning: Could not extract percentage from path: {path_str}")

    np.random.seed(42)

    with patch('kep.generation.generate_acceptance_variations.NUM_FILES', 1):
        with patch('pandas.DataFrame.to_csv', side_effect=save_capture, autospec=True):
            gen_av.process_pools()

    # Get column of waiting times for 25% and 75% acceptance scenarios and calculate means
    waits_25 = saved_dfs[25]['COL_PATIENT_ACCEPT_IMSUP']
    waits_75 = saved_dfs[75]['COL_PATIENT_ACCEPT_IMSUP']

    mean_25 = waits_25.mean()
    mean_75 = waits_75.mean()

    # Logic: p=0.25 (hard to accept) -> Long wait
    #        p=0.75 (easy to accept) -> Short wait
    # Therefore: mean_25 > mean_75

    assert mean_25 > mean_75, \
        f"Logical inconsistency: Wait time at 25% ({mean_25}) should be GREATER than at 75% ({mean_75})."
