import csv
import os

import pytest

from kep.simulation import simulator

# Mock Solver
def mock_solver_fixed_match(itG, data, max_cycle, max_chain, state):
    """
    Simulates a solver that finds a cycle between nodes 0 and 1,
    BUT ONLY if there are compatible edges (0->1 and 1->0) in the graph.
    """
    if len(itG.verts) >= 2:
        # itG.adjList is a list of lists: adjList[0] contains the neighbors of 0
        has_edge_0_1 = 1 in itG.adjList[0]
        has_edge_1_0 = 0 in itG.adjList[1]

        if has_edge_0_1 and has_edge_1_0:
            # Statistics: [NumVars, Transplants, Incompat, Unsolved, Cycles, Chains]
            stats = [len(itG.verts), 2, 2, 0, 1, 0]
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
        # Matchable pairs (0 and 1)
        ["0", "0", "A", "30", "0", "A", "0.0", "-1", "30", "0", "100", "pair", "1"],
        ["1", "1", "B", "30", "1", "B", "0.0", "-1", "30", "0", "100", "pair", "1"],
        # Compatible_Left (Compatible pair that leaves on day 5)
        ["2", "2", "O", "30", "2", "O", "0.0", "-1", "30", "0", "5", "pair", "3"],
        # Immuno (Comp=2, Accept=0) immediately immuno
        ["3", "3", "A", "30", "3", "A", "0.0", "0", "30", "0", "100", "pair", "2"],
        # Unmatched End (Leftover at end of simulation)
        ["4", "4", "AB","30", "4", "AB", "0.0", "-1", "30", "0", "10000", "pair", "1"],
        # Timeout (Leaves on the day of the match, day 30)
        ["5", "5", "A","30", "5", "AB","0.0", "-1", "30", "0", "30", "pair", "1"],
        # Immuno (Comp=2, Accept=1) wait 1 epoch then immuno
        ["6", "6", "A", "30", "6", "AB","0.0", "1", "30", "0", "1000", "pair", "2"],
        # Compatible_Left (Leaves on the day of the match, day 30)
        ["7", "7", "O", "30", "7", "O", "0.0", "-1", "30", "0", "30", "pair", "3"],
        # Timeout (Comp=2) (Leaves on the day of the match, day 30)
        ["8", "8", "A", "30", "8", "AB","0.0", "5", "30", "0", "30", "pair", "2"],
        # Timeout (Leaves on day 5)
        ["9", "9", "O", "30", "9", "O", "0.0", "-1", "30", "0", "5", "pair", "1"]
    ]

    with open(paths["char_path"], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)

    # Graph: 1->2 and 2->1
    graph_data = [
        "2, 2\n",
        "0, 1, 1\n",
        "1, 0, 1\n"
    ]

    with open(paths["graph_path"], 'w') as f:
        f.writelines(graph_data)

    return paths

def test_simulator_execution(simulation_env):
    """
    Test the main simulator execution with the mock solver.
    """

    simulyears = 1
    t_period = 30
    max_cycle = 3
    max_chain = 3
    instance_number = 1
    simulation_name = "pytest_sim"

    simulator(
        simulyears,
        str(simulation_env["char_path"]),
        str(simulation_env["graph_path"]),
        t_period,
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
    with open(simulation_env["matched_file"], 'r') as f:
        content = f.read()
        assert "1:0,1" in content, "Pairs 0 and 1 should be matched."

    # History File
    assert os.path.exists(simulation_env["history_file"]), "History file should be created."

    with open(simulation_env["history_file"], "r") as f:
        lines = f.readlines()
        header = lines[0]
        data_rows = lines[1:]

        assert len(data_rows) == 10, "History should have 10 rows (one per pair)"

        # Parse rows to dict for easier checking
        hist_data = {}
        history_epochs = {}
        for row in data_rows:
            parts = row.strip().split(",")
            pid = int(parts[0])
            status = parts[4] # Status is 5th column
            ep = int(parts[7]) # Epoch is 8th column
            hist_data[pid] = status
            history_epochs[pid] = ep

        # Pairs Status
        assert hist_data[0] == "MATCHED", "Pair 0 should be MATCHED."
        assert hist_data[1] == "MATCHED", "Pair 1 should be MATCHED."
        assert hist_data[2] == "COMPATIBLE_LEFT", "Pair 2 should be COMPATIBLE_LEFT."
        assert hist_data[3] == "IMMUNO", "Pair 3 should be IMMUNO."
        assert hist_data[4] == "UNMATCHED_END", "Pair 4 should be UNMATCHED_END because it was leftover at the end of the simulation."
        assert hist_data[5] == "TIMEOUT", "Pair 5 should be TIMEOUT (lefts on day of match)."
        assert hist_data[6] == "IMMUNO", "Pair 6 should be IMMUNO (waits 1 epoch then immuno)."
        assert hist_data[7] == "COMPATIBLE_LEFT", "Pair 7 should be COMPATIBLE_LEFT (leaves on day of match)."
        assert hist_data[8] == "TIMEOUT", "Pair 8 should be TIMEOUT (leaves on day of match)."
        assert hist_data[9] == "TIMEOUT", "Pair 9 should be TIMEOUT (leaves on day 5)."
        # Pairs Epoch
        assert history_epochs[0] == 1, "Pair 0 should be matched in Epoch 1"
        assert history_epochs[1] == 1, "Pair 1 should be matched in Epoch 1"
        assert history_epochs[2] == 1, "Pair 2 should timeout in Epoch 1, since it departs at day 5 and t=30."
        assert history_epochs[3] == 1, "Pair 3 should be Immuno in Epoch 1"
        # 360/30 = 12 epochs in total
        assert history_epochs[4] == 12, "Pair 4 should remain until Epoch 12 (End of Simulation)"

    # Results File
    assert os.path.exists(simulation_env["results_file"]), "Results file should be created."
    with open(simulation_env["results_file"], "r") as f:
        lines = f.readlines()

        # The simulator runs for 12 epochs, so we expect 12 data lines plus header.
        assert len(lines) == 12, f"O ficheiro de resultados devia ter 12 linhas, mas tem {len(lines)}."

        content = lines[0].strip().split(", ")

        assert content[0] == "1"
        assert content[1] == "30"
        assert content[4].strip() == simulation_name, "Simulation name should match."
        assert content[5].strip() == "1", "The first line should correspond to epoch 1."
        # 6 nodes at start but pair 2 leaves at day 5, so only 5 remain for epoch 1
        assert content[6].strip() == "8", "There should be 8 nodes in the pool at start of epoch 1."

        content_last = lines[11].strip().split(", ")

        assert content_last[0] == "1"
        assert content_last[1] == "30"
        assert content_last[4].strip() == simulation_name, "Simulation name should match."
        assert content_last[5].strip() == "12", "The last line should correspond to epoch 12."
        assert content_last[6].strip() == "1", "Pair 4 should remain until the last epoch."

def test_simulator_with_logging(simulation_env, monkeypatch, capsys):
    """
    Test the simulator execution with logging enabled.
    """
    monkeypatch.setitem(simulator.__globals__, "LOG", True)

    simulyears = 1
    t_period = 30
    max_cycle = 3
    max_chain = 3
    instance_number = 1
    simulation_name = "pytest_sim_log"

    simulator(
        simulyears,
        str(simulation_env["char_path"]),
        str(simulation_env["graph_path"]),
        t_period,
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
    assert "Starting iteration" in captured.out
    assert "Iteration graph has" in captured.out
    assert "Solver done in" in captured.out
    assert "(Half-Compat) -> accepted Immunosuppression immediately." in captured.out
    assert "(Half-Compat) waited 1 epochs" in captured.out
    assert "KEP matched pairs" in captured.out
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
    t_period = 30
    max_cycle = 3
    max_chain = 3
    instance_number = 100
    simulation_name = "pytest_sim_error"

    simulator(
        simulyears,
        str(simulation_env["char_path"]),
        str(simulation_env["graph_path"]),
        t_period,
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
