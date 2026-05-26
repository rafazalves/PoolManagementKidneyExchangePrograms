from unittest.mock import MagicMock, patch

import gurobipy as gp

import kep.policies.policy_kep_static as policy_static
from kep.compatibility_graph import Digraph


def test_priority_max_transplants(load_kep_data):
    """
    Test priority compatibility without waiting time.
    
    Scenario:
    - Cycle A: 0 -> 1 -> 4 -> 0.
      Consists of pairs 0, 1, 4.
      
    - Cycle B: 0 -> 2 -> 3 -> 0.
      Consists of pairs 0, 2, 3.

    - Cycle C: 7 -> 8 -> 7.
      Consists of pairs 7, 8.
      
    - Cycle D: 7 -> 9 -> 10 -> 7.
      Consists of pairs 7, 9, 10.

    Since in this solver we do not consider waiting time nor compatibility, it should not affect the outcome.

    Expected Result:
    Cycle A or B should be chosen, because since we do not consider waiting time nor compatibility, both cycles have equal "normal" weight (based on number of transplants). 
    Cycle D has a higher weight (based on number of transplants (3)) than Cycle C (2) and should be chosen, too.

    - Chain A: 5 -> 6.
    The solver also returns a chain with an altruistic donor (pair 5) donating to pair 6.
    """

    # Load the files for the test
    digraph, data = load_kep_data("pool5_test_solver.csv", "arc5_test_solver.txt")

    # Execute the solver
    outputlist, selected_structures = policy_static.solver(digraph, data, max_cycle=3, max_chain=3, state=None)

    print(f"\n[DEBUG] Selected structures: {selected_structures}")

    assert len(outputlist) == 6, "The outputlist should have exactly 6 elements."
    assert outputlist[0] == 11, f"Should have 11 nodes, but got {outputlist[0]}."

    # There should be exactly 2 cycles chosen and 1 chain
    assert len(selected_structures) == 3, "Should have chosen 2 cycles due to conflict on node 0 and node 7, and 1 chain (5->6)."

    assert outputlist[4] == 2, "Should have exactly 2 cycles selected."
    assert outputlist[5] == 1, "Should have exactly 1 chain selected."

    cycle1 = selected_structures[0]

    # Check if the nodes of the winning cycle are exactly {0, 2, 3} or {0, 1, 4} depending on gurobi's choice. Normally it will choose the one with the higher index first but it can vary.
    assert set(cycle1) == {0, 1, 4}, \
        f"The solver chose cycle {cycle1} instead of cycle [0, 1, 4]."

    cycle2 = selected_structures[1]

    # Check if the nodes of the winning cycle are exactly {7, 9, 10}
    assert set(cycle2) == {7, 9, 10}, \
        f"Priority Error: The solver chose cycle {cycle2} instead of cycle [7, 9, 10] (which has more transplants)."

    chain = selected_structures[2]

    # Check if the chain is exactly [5, 6]
    assert chain == [5, 6], "Priority Error: The solver chose chain {chain} instead of chain [5, 6]."

    # Since the cycle [0, 1, 4] has 2 incompatible pairs plus the pair 6 from the chain, the total should be 3.
    assert outputlist[2] == 3, f"Expected to count 3 incompatible pairs, but counted {outputlist[2]}."

def test_solver_empty_graph():
    """Tests if the solver handles an empty graph well."""
    g = Digraph()
    g.nv = 0
    g.verts = []

    data = {}

    results, structures = policy_static.solver(g, data, 2, 2, {})

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
    digraph, data = load_kep_data("pool5_test_solver.csv", "arc5_test_solver.txt")

    # Make LOG True and REVERSE 1
    monkeypatch.setattr(policy_static, "LOG", True)
    monkeypatch.setattr(policy_static, "REVERSE", 1)

    # 2. Caminho COMPLETO para o patch funcionar dentro da pasta kep/solvers
    with patch("kep.policies.policy_kep_static.gp.Model") as MockModel:
        mock_instance = configure_mock_gurobi(MockModel)
        mock_instance.status = gp.GRB.OPTIMAL

        policy_static.solver(digraph, data, max_cycle=3, max_chain=3, state=None)

    captured = capsys.readouterr()
    assert "Generating cycles..." in captured.out
    assert "Generating chains..." in captured.out
    assert "Optimization finished. Total Transplants:" in captured.out

def test_gurobi_infeasible(load_kep_data, monkeypatch, capsys):
    '''
    Test Gurobi infeasibible status handling.
    '''
    monkeypatch.setattr(policy_static, "LOG", True)
    digraph, data = load_kep_data("pool5_test_solver.csv", "arc5_test_solver.txt")

    with patch("kep.policies.policy_kep_static.gp.Model") as MockModel:
        mock_instance = configure_mock_gurobi(MockModel)

        mock_instance.status = gp.GRB.INFEASIBLE
        policy_static.solver(digraph, data, max_cycle=3, max_chain=3, state=None)
        captured = capsys.readouterr()
        assert "Model Infeasible!" in captured.out
