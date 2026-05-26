import csv
import os

import pytest

from kep.constants import *
from kep.simulation import simulator_static


# Mock Solver
def mock_solver_fixed_match(itG, data, max_cycle, max_chain, state):
    """
    Simulate a solver that always finds a cycle between the first two nodes of the current graph, if there are at least 2 nodes.
    """
    if len(itG.verts) >= 2:
        # Statistics: [NumVars, Transplants, Incompat, Unsolved, Cycles, Chains]
        stats = [len(itG.verts), 2, 2, 0, 1, 0]
        # Cycle between index 0 and index 1 (which will correspond to pairs 0 and 1)
        cycles = [[0, 1]]
        return stats, cycles
    return [0, 0, 0, 0, 0, 0], []

# Simulation Configuration
@pytest.fixture
def simulation_env(tmp_path):
    paths = {
        "char_path": tmp_path / "test_pool.csv",
        "graph_path": tmp_path / "test_graph.txt",
        "results_file": tmp_path / "results.csv",
        "matched_file": tmp_path / "matched.csv",
        "history_file": tmp_path / "history.csv",
        "error_file": tmp_path / "error.log"
    }

    # Header for CSV file
    header = [
        "COL_PAIR_ID", "COL_DONOR_ID", "COL_DONOR_BLOOD", "COL_DONOR_AGE",
        "COL_PATIENT_ID", "COL_PATIENT_BLOOD", "COL_PATIENT_PRA", "COL_PATIENT_ACCEPT_IMSUP",
        "COL_PATIENT_AGE", "COL_PAIR_ARRIVAL", "COL_PAIR_DEPARTURE", "COL_PAIR_DESCRIPTION",
        "COL_PAIR_COMPATIBILITY"
    ]

    csv_data = [
        # Matchable pairs (0 and 1). Pair 1 leaves at day 200 and pair 0 arrives at day 500
        # but since we are in a static model, they can match.
        ["0", "0", "A", "30", "0", "A", "0.0", "-1", "30", "500", "1000", "pair", "1"],
        ["1", "1", "B", "30", "1", "B", "0.0", "-1", "30", "0", "200", "pair", "1"],
        # Unmatch pair 2
        ["2", "2", "O", "30", "2", "O", "0.0", "-1", "30", "0", "850", "pair", "3"],
        # Immuno pair 3
        ["3", "3", "O", "30", "3", "O", "0.0", "1", "30", "100", "500", "pair", "2"],
        # Unmatch pair 4 because did not accept immunosupression
        ["4", "4", "O", "30", "4", "O", "0.0", "-1", "30", "300", "500", "pair", "2"]
    ]

    with open(paths["char_path"], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)

    # Graph: 1->2 and 2->1
    graph_data = [
        "3, 3\n",
        "0, 1, 1\n",
        "1, 0, 1\n",
        "2, 0, 1\n"
    ]

    with open(paths["graph_path"], 'w') as f:
        f.writelines(graph_data)

    return paths

def test_simulator_static_execution(simulation_env):
    """
    Test the simulator_static execution with the mock solver.
    """

    simulyears = 1
    max_cycle = 3
    max_chain = 3
    instance_number = 1
    simulation_name = "TestStatic"

    # Executar Simulator Static com Mock Solver
    simulator_static(
        simulyears,
        str(simulation_env["char_path"]),
        str(simulation_env["graph_path"]),
        max_cycle,
        max_chain,
        mock_solver_fixed_match,
        str(simulation_env["results_file"]),
        str(simulation_env["matched_file"]),
        str(simulation_env["history_file"]),
        str(simulation_env["error_file"]),
        instance_number,
        simulation_name
    )

    # Matches
    assert os.path.exists(simulation_env["matched_file"])
    with open(simulation_env["matched_file"], "r") as f:
        content = f.read().strip()
        assert "1:0,1" in content or "1:1,0" in content, "Matched file should contain the matched cycle between pairs 0 and 1."

    # History File
    assert os.path.exists(simulation_env["history_file"]), "History file should be created."

    with open(simulation_env["history_file"], "r") as f:
        lines = f.readlines()
        header = lines[0]
        data_rows = lines[1:]

        assert len(data_rows) == 5, "History should have 5 rows (one per pair)"

        # Parse rows to dict for easier checking
        hist_data = {}
        for row in data_rows:
            parts = row.strip().split(",")
            pid = int(parts[0])
            status = parts[4]
            hist_data[pid] = status

        assert hist_data[0] == "MATCHED", "Pair 0 should be MATCHED."
        assert hist_data[1] == "MATCHED", "Pair 1 should be MATCHED."
        assert hist_data[2] == "COMPATIBLE_LEFT", "Pair 2 should be COMPATIBLE_LEFT."
        assert hist_data[3] == "IMMUNO", "Pair 3 should be IMMUNO."
        assert hist_data[4] == "UNMATCHED_END", "Pair 4 should be UNMATCHED since it did not accept immunosuppression."

    # Results File
    assert os.path.exists(simulation_env["results_file"]), "Results file should be created."
    with open(simulation_env["results_file"], "r") as f:
        lines = f.readlines()

        # The simulator_static only runs one epoch, so we expect one data line plus header.
        assert len(lines) == 1, f"O ficheiro de resultados devia ter 1 linha, mas tem {len(lines)}."

        content = lines[0].strip().split(", ")

        assert content[0] == "1"
        assert content[1] == "STATIC"
        assert content[5].strip() == "1", "There should be 1 epoch."
        assert content[6].strip() == "5", "All 5 nodes should be in the static pool."

def test_simulator_static_with_logging(simulation_env, monkeypatch, capsys):
    """
    Test the simulator_static execution with logging enabled.
    """
    monkeypatch.setitem(simulator_static.__globals__, "LOG", True)

    simulyears = 1
    max_cycle = 3
    max_chain = 3
    instance_number = 1
    simulation_name = "pytest_sim_log"

    simulator_static(
        simulyears,
        str(simulation_env["char_path"]),
        str(simulation_env["graph_path"]),
        max_cycle,
        max_chain,
        mock_solver_fixed_match,
        str(simulation_env["results_file"]),
        str(simulation_env["matched_file"]),
        str(simulation_env["history_file"]),
        str(simulation_env["error_file"]),
        instance_number,
        simulation_name
    )

    captured = capsys.readouterr()
    assert "Pool size" in captured.out
    assert "Graph constructed" in captured.out
    assert "Solver done in" in captured.out
    assert "Iteration done in" in captured.out
    assert "pairs had immuno-transplants" in captured.out
    # Basic check to ensure it ran to completion
    assert os.path.exists(simulation_env["results_file"])

def test_solver_error_handling(simulation_env, capsys):
    """
    Test the scenario where the solver fails.
    """

    # Mock Solver that simulates failure
    def mock_solver_failure(itG, data, max_cycle, max_chain, state):
        return -1, [] # Return error code

    simulyears = 1
    max_cycle = 3
    max_chain = 3
    instance_number = 100
    simulation_name = "pytest_sim_error"

    simulator_static(
        simulyears,
        str(simulation_env["char_path"]),
        str(simulation_env["graph_path"]),
        max_cycle,
        max_chain,
        mock_solver_failure, # INJECTION OF THE SOLVER WITH ERROR
        str(simulation_env["results_file"]),
        str(simulation_env["matched_file"]),
        str(simulation_env["history_file"]),
        str(simulation_env["error_file"]),
        instance_number,
        simulation_name
    )

    # Check if the error file was created
    assert os.path.exists(simulation_env["error_file"])

    captured = capsys.readouterr()
    assert "Error detected: Abort." in captured.out

    with open(simulation_env["error_file"], "r") as f:
        content = f.read().strip()
        # The simulator writes: (instance_number, len(cycles))
        # instance_number is 100, len(cycles) is 0
        expected_error = "100 0"
        assert content == expected_error
