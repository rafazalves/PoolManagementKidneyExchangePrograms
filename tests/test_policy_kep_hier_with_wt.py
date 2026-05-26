from unittest.mock import MagicMock, patch

import gurobipy as gp
import pytest

import kep.policies.policy_kep_hier_with_wt as policy_hier_with_wt
from kep.compatibility_graph import Digraph


def test_priority_compatibility_with_wt(load_kep_data):
    """
    Test priority compatibility with waiting time.
    
    Scenario:
    - Cycle A: 0 -> 1 -> 4 -> 0.
      Consists of pairs 0, 1, 4.
      
    - Cycle B: 0 -> 2 -> 3 -> 0.
      Consists of pairs 0, 2, 3.

    All pairs have COMPATIBILITY=1 (Incompatible).
    All pairs have the same waiting time, except pair 2 which has been waiting longer.

    Expected Result:
    Cycle B has a higher weight and should be chosen, because if all pairs have same compatibility and pair 2 has a longer waiting time, giving Cycle B an edge in the lexicographic optimization.
    """

    # Load the files for the test
    digraph, data = load_kep_data("pool1_test_solver.csv", "arc1_test_solver.txt")

    # Execute the solver
    outputlist, selected_structures = policy_hier_with_wt.solver(digraph, data, max_cycle=3, max_chain=3, state=None)

    print(f"\n[DEBUG] Selected structures: {selected_structures}")

    assert len(outputlist) == 6, "The outputlist should have exactly 6 elements."
    assert outputlist[0] == 7, f"Should have 7 nodes, but got {outputlist[0]}."

    # There should be exactly 1 cycle chosen (because they share node 0, they are mutually exclusive)
    assert len(selected_structures) == 1, "Should have chosen only 1 cycle due to conflict on node 0."

    assert outputlist[4] == 1, "Should have exactly 1 cycle selected."
    assert outputlist[5] == 0, "Should have exactly 0 chains selected."

    cycle = selected_structures[0]

    # Check if the nodes of the winning cycle are exactly {0, 2, 3}
    assert set(cycle) == {0, 2, 3}, \
        f"Priority Error: The solver chose cycle {cycle} instead of cycle [0, 2, 3]."

    # Since the cycle [0, 2, 3] has 3 incompatible pairs, the total should be 3.
    assert outputlist[2] == 3, f"Expected to count 3 incompatible pairs, but counted {outputlist[2]}."

def test_priority_incompatible_vs_half(load_kep_data):
    """  
    Test priority compatibility "better" than waiting time.
    
    Scenario:
    - Cycle A: 0 -> 1 -> 4 -> 0.
      Consists of pairs 0, 1, 4.
      
    - Cycle B: 0 -> 2 -> 3 -> 0.
      Consists of pairs 0, 2, 3.

    All pairs have COMPATIBILITY=1 (Incompatible) except pair 2 which has COMPATIBILITY=2 (Half-Compatible).
    All pairs have the same waiting time, except pair 2 which has been waiting longer.

    Expected Result:
    Cycle A has a higher weight and should be chosen, because even if pair 2 has a longer waiting time, its half-compatibility makes Cycle B less favorable than Cycle A, which has all incompatible pairs.

    - Chain A: 5 -> 6.
    - Chain B: 5 -> 7.
    The solver also returns a chain with an altruistic donor (pair 5) donating to pair 6 because pair 6 is incompatible (COMPATIBILITY=1) while pair 7 is half-compatible (COMPATIBILITY=2).
    """

    # Load the files for the test
    digraph, data = load_kep_data("pool2_test_solver.csv", "arc2_test_solver.txt")

    # Execute the solver
    outputlist, selected_structures = policy_hier_with_wt.solver(digraph, data, max_cycle=3, max_chain=3, state=None)

    print(f"\n[DEBUG] Selected structures: {selected_structures}")

    assert len(outputlist) == 6, "The outputlist should have exactly 6 elements."
    assert outputlist[0] == 8, f"Should have 8 nodes, but got {outputlist[0]}."

    # There should be exactly 1 cycle chosen (because they share node 0, they are mutually exclusive) and 1 chain (5->6)
    assert len(selected_structures) == 2, "Should have chosen only 1 cycle due to conflict on node 0, and 1 chain (5->6)."

    assert outputlist[4] == 1, "Should have exactly 1 cycle selected."
    assert outputlist[5] == 1, "Should have exactly 1 chain selected."

    cycle = selected_structures[0]

    # Check if the nodes of the winning cycle are exactly {0, 1, 4}
    assert set(cycle) == {0, 1, 4}, \
        f"Priority Error: The solver chose cycle {cycle} instead of cycle [0, 1, 4] (which has more incompatible pairs)."

    chain = selected_structures[1]

    # Check if the chain is exactly [5, 6]
    assert chain == [5, 6], "Priority Error: The solver chose chain {chain} instead of chain [5, 6]."

    # Since the cycle [0, 1, 4] has 3 incompatible pairs plus the pair 6 from the chain, the total should be 4.
    assert outputlist[2] == 4, f"Expected to count 4 incompatible pairs, but counted {outputlist[2]}."

def test_solver_empty_graph():
    """Tests if the solver handles an empty graph well."""
    g = Digraph()
    g.nv = 0
    g.verts = []

    data = {}

    results, structures = policy_hier_with_wt.solver(g, data, 2, 2, {})

    # Should return zeros and empty list
    assert results == [0, 0, 0, 0, 0, 0], "Expected all zero results for empty graph."
    assert structures == [], "Expected empty structures list for empty graph."

# Function to configure the Gurobi mock
def configure_mock_gurobi(mock_model_class):
    """
    Configure the Gurobi mock to simulate model behavior.
    """
    mock_instance = mock_model_class.return_value
    mock_variable = MagicMock()
    mock_variable.X = 0.0

    mock_instance.addVar.return_value = mock_variable
    mock_instance.addVars.return_value.__getitem__.return_value = mock_variable
    mock_instance.getVars.return_value = [mock_variable]

    return mock_instance

def test_logs_and_reverse_execution(load_kep_data, monkeypatch, capsys):
    digraph, data = load_kep_data("pool2_test_solver.csv", "arc2_test_solver.txt")

    # Make LOG True and REVERSE 1
    monkeypatch.setattr(policy_hier_with_wt, "LOG", True)
    monkeypatch.setattr(policy_hier_with_wt, "REVERSE", 1)

    with patch("kep.policies.policy_kep_hier_with_wt.gp.Model") as MockModel:
        mock_instance = configure_mock_gurobi(MockModel)
        mock_instance.status = gp.GRB.OPTIMAL

        def side_effect_add_var(*args, **kwargs):
            name = kwargs.get('name', '')
            mock_var = MagicMock()
            mock_var.VarName = name

            # Force cycle 0 and chain 0 to be selected
            if name == "c_0" or name == "s_0":
                mock_var.X = 1.0
            else:
                mock_var.X = 0.0
            return mock_var

        mock_instance.addVar.side_effect = side_effect_add_var
        mock_instance.addVars.return_value = {i: MagicMock(X=0.0) for i in range(100)}

        policy_hier_with_wt.solver(digraph, data, max_cycle=3, max_chain=3, state=None)


    captured = capsys.readouterr()

    # Log verifications
    assert "Generating cycles..." in captured.out
    assert "Generating chains..." in captured.out

    # Check if cycle 0 and chain 0 were selected
    assert " > Selected Cycle ID 0:" in captured.out
    assert " > Selected Chain ID 0:" in captured.out

def test_gurobi_infeasible(load_kep_data, monkeypatch, capsys):
    '''
    Test Gurobi infeasibible status handling.
    '''
    digraph, data = load_kep_data("pool2_test_solver.csv", "arc2_test_solver.txt")

    with patch("kep.policies.policy_kep_hier_with_wt.gp.Model") as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.status = gp.GRB.INFEASIBLE
        mock_instance.addVars.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="Infeasible at lex level"):
            policy_hier_with_wt.solver(digraph, data, max_cycle=3, max_chain=3, state=None)

def test_gurobi_timelimit(load_kep_data):
    """
    Test Gurobi status other than OPTIMAL or INFEASIBLE (in this test: TIME_LIMIT).
    This hits the 'continue' statement.
    """
    digraph, data = load_kep_data("pool2_test_solver.csv", "arc2_test_solver.txt")

    with patch("kep.policies.policy_kep_hier_with_wt.gp.Model") as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.status = gp.GRB.TIME_LIMIT

        # Mock addVars so that accessing x[i].X does not fail
        mock_variable = MagicMock()
        mock_variable.X = 0.0
        mock_variable.VarName = "c_0"
        mock_instance.addVar.return_value = mock_variable

        outputlist, selected_structures = policy_hier_with_wt.solver(digraph, data, max_cycle=3, max_chain=3, state=None)

        assert outputlist[3] > 0, "Should have counted unsolved problems due to TIME_LIMIT."

        # The code should continue execution and return a valid list
        assert len(outputlist) == 6, "The outputlist should have exactly 6 elements."
