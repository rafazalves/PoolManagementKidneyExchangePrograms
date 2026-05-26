import io
import os

import pytest

from kep.constants import *
from kep.utils import log_pair_history, read_characterization, read_graph, read_times


@pytest.fixture
def pool_file():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_dir, "tests_data", "pool_test_utils.csv")

@pytest.fixture
def temp_graph_file(tmp_path):
    """Creates a temporary graph file for testing."""
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_graph.txt"
    content = "3, 4\n0,1,1\n1,0,1\n1,2,1\n2,0,1\n"
    p.write_text(content)
    return str(p)

def test_read_char_types(pool_file):
    """Tests if data types (float, int, bool) are correctly converted."""
    # Execute function to read characterization csv files
    char_dict = read_characterization(pool_file)

    # Pair 0
    p0 = char_dict[0]
    assert isinstance(p0[COL_PATIENT_PRA], float), "PRA should be float."
    assert p0[COL_PATIENT_PRA] == 0.55, "PRA value incorrect."
    assert p0[COL_PATIENT_ACCEPT_IMSUP], "String '1' should be converted to True."
    assert p0[COL_PAIR_COMPATIBILITY] == 1, "Compatibility should be int."

    # Pair 1
    p1 = char_dict[1]
    assert isinstance(p1[COL_PAIR_ID], int), "ID should be int."
    assert not p1[COL_PATIENT_ACCEPT_IMSUP], "String '0' should be converted to False."
    assert p1[COL_PAIR_ARRIVAL] == 10.5, "Arrival should support floats."

def test_read_malformed_csv(tmp_path, capsys):
    """
    Covers lines 15 (empty rows) and 18-19 (short rows).
    """
    d = tmp_path / "data"
    d.mkdir()
    f = d / "malformed_pool.csv"

    # Create CSV content:
    # 1. Header
    # 2. Empty row
    # 3. Short row (len < 13)
    # 4. Valid row (to ensure function finishes successfully)

    valid_row = "0,0,O,30,0,A,0.1,1,30,0.0,100.0,pair,3"

    content = (
        "header,line\n"          # Header
        "\n"                     # Empty row (Line 15)
        " , , \n"                # Whitespace row (Line 15)
        "0,1,2\n"                # Short row (Line 18-19)
        f"{valid_row}\n"         # Valid row
    )

    f.write_text(content)

    result = read_characterization(str(f))

    # Assert valid row was parsed
    assert len(result) == 1
    assert int(result[0][COL_PAIR_ID]) == 0

    # Verify warnings were printed
    captured = capsys.readouterr()
    assert "Warning: Skipping empty row" in captured.out
    assert "Warning: Skipping malformed row" in captured.out

def test_read_char_descriptions(pool_file):
    """Tests if the description (pair vs altruist) is normalized."""
    char_dict = read_characterization(pool_file)

    assert char_dict[5][COL_PAIR_DESCRIPTION] == 'altruist', "Should convert to lowercase."
    assert char_dict[6][COL_PAIR_DESCRIPTION] == 'pair', "Should have no spaces and convert to lowercase."

def test_read_graph(temp_graph_file):
    """Tests reading the graph structure"""
    verts, arcs = read_graph(temp_graph_file)

    # Check Vertices
    assert len(verts) == 3, "Should have 3 vertices."
    assert 0 in verts
    assert 2 in verts

    # Check Arcs
    assert len(arcs) == 4, "Should have 4 arcs."
    assert (0, 1) in arcs, "Arc (0,1) missing."
    assert (0, 2) not in arcs, "Arc (0,2) should not exist."
    assert (2, 0) in arcs, "Arc (2,0) missing."

def test_read_times(pool_file):
    """Tests time extraction and Min/Max calculation."""
    charact = read_characterization(pool_file)
    times, min_arr, max_dep = read_times(charact)

    # Check specific pair times (Pair 1: Arrival 10.5, Departure 200)
    assert times[1] == (10.5, 200.0), "Pair 1 times incorrect."

    # Check Pair 0 (Arrival 0, Departure 200)
    assert times[0] == (0.0, 200.0), "Pair 0 times incorrect."

    # Check Global Min/Max
    assert min_arr == 0.0, "Minimum arrival should be 0."
    assert max_dep == 200.0, "Maximum departure should be 200."

def test_log_pair_history(pool_file):
    """Tests if the log string is formatted correctly using a mock file."""

    charact = read_characterization(pool_file)
    mock_times = {
        0: (10.0, 500.0),
        2: (20.0, 600.0),
        5: (30.0, 700.0)
    }
    mock_file = io.StringIO()

    # Define test parameters for Pair 0 (Medium PRA)
    pair_id = 0
    departure_time = 100.0
    status = "MATCHED"
    epoch = 5

    # Run Function
    log_pair_history(
        pair_id,
        departure_time,
        status,
        mock_times,
        charact,
        mock_file,
        epoch
    )

    lines = mock_file.getvalue().strip().split('\n')
    parts = lines[0].split(',')

    # Expected Format:
    # id, arrival, departure, wait, status, match_type, match_group, epoch, p_blood, d_blood, pra_cat, comp_type, description
    assert parts[0] == "0", "Pair ID mismatch"
    assert parts[1] == "10.0", "Arrival time mismatch (from times dict)"
    assert parts[2] == "100.0", "Departure time mismatch (arg)"
    assert parts[3] == "90.0", "Wait time calculation error (100 - 10)"
    assert parts[4] == "MATCHED", "Status mismatch"
    assert parts[7] == "5", "Epoch mismatch"

    # Verify Characterization Data extraction (From CSV)
    # P_Blood=A, D_Blood=O, PRA=0.55 (Medium)
    assert parts[8] == "A", "Patient blood mismatch"
    assert parts[9] == "O", "Donor blood mismatch"
    assert parts[10] == "Medium", "PRA Category logic error (0.55 should be Medium)"

    # Pair 2 and Pair 5 tests
    # Pair 2 in CSV has PRA 0.90 -> pra_category should be "High"
    log_pair_history(2, 200.0, "MATCHED", mock_times, charact, mock_file, 1)

    lines = mock_file.getvalue().strip().split('\n')
    parts_high = lines[1].split(',')

    assert parts_high[0] == "2", "Pair ID 2 mismatch"
    assert parts_high[10] == "High", "PRA > 0.8 should be categorized as 'High'"

    # Pair 5 in CSV is altruist -> pra_category should be "NA"
    log_pair_history(5, 300.0, "MATCHED", mock_times, charact, mock_file, 1)

    lines = mock_file.getvalue().strip().split('\n')
    parts_altruist = lines[2].split(',')

    assert parts_altruist[0] == "5", "Pair ID 5 mismatch"
    assert parts_altruist[10] == "NA", "Altruist description should result in PRA category 'NA'"

def test_log_pair_history_errors(pool_file):
    """
    Tests error handling in log_pair_history:
    - ValueError (Invalid PRA string)
    - IndexError (Missing columns)
    """
    charact = read_characterization(pool_file)
    mock_times = {0: (10.0, 100.0)}
    mock_file = io.StringIO()

    # ValueError
    charact[0][COL_PATIENT_PRA] = "invalid_float" # Inject bad data

    log_pair_history(0, 100.0, "MATCHED", mock_times, charact, mock_file, 1)
    output_val = mock_file.getvalue().strip().split('\n')[-1]
    assert "ERR_VAL" in output_val, "Should handle ValueError gracefully"

    # IndexError
    broken_charact = {99: []} # Pair 99 has empty data
    broken_times = {99: (0.0, 0.0)}

    log_pair_history(99, 100.0, "MATCHED", broken_times, broken_charact, mock_file, 1)
    output_idx = mock_file.getvalue().strip().split('\n')[-1]
    assert "ERR" in output_idx, "Should handle IndexError gracefully"
