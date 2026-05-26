
from kep.constants import *
from kep.utils import (
        create_cf_constraints,
        create_global_compatibility_coefs,
        create_global_wtime_coefs,
)


def create_dummy_char(n, altruists_indices):
        """
        Helper function to create a dummy characterization dictionary to test Description field.
        """
        char = {}
        for i in range(n):
            # The max index used in the tests is 12
            row = [''] * 12

            if i in altruists_indices:
                row[COL_PAIR_DESCRIPTION] = 'altruist'
            else:
                row[COL_PAIR_DESCRIPTION] = 'pair'

            char[i] = row
        return char

def test_create_cf_constraints_basic():
    n_vertices = 4
    # Cycle 0: Nodes 0 and 1
    cycles = [[0, 1]]
    # Chain 0: Nodes 2 and 3
    chains = [[2, 3]]

    # The function should map:
    # Variable 0 -> Cycle 0
    # Variable 1 -> Chain 0 (because it comes after cycles)
    constraints = create_cf_constraints(n_vertices, cycles, chains)

    # Node 0 participates in structure index 0 (Cycle)
    assert constraints[0] == [0]
    # Node 1 participates in structure index 0 (Cycle) but not in structure index 1 (Chain)
    assert constraints[1] == [0]
    assert constraints[1] != [1]
    # Node 2 participates in structure index 1 (Chain)
    assert constraints[2] == [1]

def test_create_cf_constraints_overlap():
    """
    Tests if a node that appears in multiple structures generates the correct constraints.
    """
    n_vertices = 5

    # Cycle 0: [0, 1]
    # Cycle 1: [1, 2]
    cycles = [[0, 1], [1, 2]]

    # Chain 0: [3, 4]
    # Chain 1: [4, 0]
    chains = [[3, 4], [4, 0]] # Chain indices 0 and 1 -> Global indices 2 and 3

    constraints = create_cf_constraints(n_vertices, cycles, chains)

    # Check Node 1 (appears in Cycles 0 and 1)
    assert sorted(constraints[1]) == [0, 1], "Node 1 should be constrained by cycles 0 and 1."

    # Check Node 0 (appears in Cycle 0 and Chain 1 -> global index 3)
    assert sorted(constraints[0]) == [0, 3], "Node 0 should be constrained by cycle 0 and chain 1 (index 3)."

    # Check Node 4 (appears in Chains 0 and 1 -> global index 2 and 3)
    assert sorted(constraints[4]) == [2, 3], "Node 4 should be constrained by chains 0 and 1 (index 2 and 3)."

def test_create_global_wtime_coefs():
    """
    Test the calculation of waiting time coefficients and altruist filtering.
    """
    # Cycle 0: [0, 1] (Normal Pairs)
    # Chain 0: [2, 3] (Node 2 is Altruist, Node 3 is Pair)
    cycles = [[0, 1]]
    chains = [[2, 3]]
    wp_pairs = [10, 5, 10, 10] # Node 2 is Altruist with WT=10, should be ignored

    # Mark node 2 as altruist
    char = create_dummy_char(4, altruists_indices=[2])

    # Execute function
    lexcoefs = create_global_wtime_coefs(0, cycles, chains, wp_pairs, char)

    # Check Keys (Should be sorted descending: 10, 5)
    assert sorted(list(lexcoefs.keys()), reverse=True) == [10, 5]

    # WT=10: Cycle0(Node0=10) -> 1. Chain0(Node2=Altruist_Ignored, Node3=10) -> 1.
    assert lexcoefs[10] == [1, 1], "Error in coefficients for WT=10. Check if altruist was ignored."

    # WT=5: Cycle0(Node1=5) -> 1. Chain0(None) -> 0.
    assert lexcoefs[5] == [1, 0], "Error in coefficients for WT=5."

def test_create_global_wtime_coefs_altruist_only():
    """
    Tests if a waiting time present ONLY in an altruist results in None.
    This happens because the sum of coefficients would be 0.
    """
    cycles = []
    chains = [[0, 1]] # 0 is Altruist, 1 is Pair

    wp_pairs = [100, 50] # Altruist has 'wait time' 100, Pair has 50
    char = create_dummy_char(2, altruists_indices=[0])

    lexcoefs = create_global_wtime_coefs(0, cycles, chains, wp_pairs, char)

    # 100 is only in altruist, should be "None"
    # lexcoefs[wt] = None if the sum is 0
    assert lexcoefs[100] is None, "WT present only in altruists should result in None."

    # 50 belongs to the pair, should have a valid coefficient
    assert lexcoefs[50] == [1]

def test_create_global_compatibility_coefs():
    """
    Tests compatibility coefficients.
    """
    cycles = [[0, 1]]
    chains = [[2, 3]]

    # Compatibility Scores
    # 0: 3
    # 1: 2
    # 2: 4 (Altruist)
    # 3: 3
    comp_pairs = [3, 2, 4, 3]
    char = create_dummy_char(4, altruists_indices=[2])

    lexcoefs = create_global_compatibility_coefs(0, cycles, chains, comp_pairs, char)

    # Check Keys: 3, 2
    assert 3 in lexcoefs
    assert 2 in lexcoefs

    # Check Comp=3
    # Cycle 0: Node 0(3) -> 1
    # Chain 0: Node 3(3) -> 1
    assert lexcoefs[3] == [1, 1], "Comp=3 should count all nodes with this score (including altruists if they have score)."

    # Check Comp=2
    # Cycle 0: Node 1(2) -> 1
    # Chain 0: Nobody -> 0
    assert lexcoefs[2] == [1, 0]
