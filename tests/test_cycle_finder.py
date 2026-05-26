import csv
import os

import pytest

from kep.compatibility_graph import CycleFinder, Digraph
from kep.constants import *


def read_char_csv(filename):
        """
        Helper function to read the CSV and convert it to the dictionary that CycleFinder expects.
        """
        char_dict = {}
        if not os.path.exists(filename):
            pytest.fail(f"Pool file not found: {filename}")
        with open(filename, "r") as f:
            reader = csv.reader(f)
            header = next(reader) # skip header

            for row in reader:
                if not row: continue

                # Map columns according to constants
                pair_id = int(row[COL_PAIR_ID])
                char_dict[pair_id] = {
                    COL_PAIR_ID: pair_id,
                    COL_DONOR_ID: int(row[COL_DONOR_ID]),
                    COL_PAIR_DESCRIPTION: row[COL_PAIR_DESCRIPTION].strip(),
                    # No other fields are needed for current tests
                }
        return char_dict

@pytest.fixture
def cf_setup():
    """Setup for CycleFinder tests. Loads the graph and characterization files."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, "tests_data")

    arc_file = os.path.join(data_dir, "arc_test_cyclefinder.txt")
    pool_file = os.path.join(data_dir, "pool_test_cyclefinder.csv")

    g = Digraph()
    if os.path.exists(arc_file):
        g.load(arc_file)
    else:
        pytest.fail(f"Graph file not found at: {arc_file}")

    characterization = read_char_csv(pool_file)
    return g, characterization

def test_initialization_altruists(cf_setup):
    """
    Verify if CycleFinder correctly identifies who are Altruists and who are Pairs.
    """
    g, characterization = cf_setup
    cf = CycleFinder(g, characterization)

    # Verify Altruists
    assert 5 in cf.altruistic_nodes, "Node 5 should be identified as altruist."
    assert len(cf.altruistic_nodes) == 1, "There should be only 1 altruist (node 5)."

    # Verify Pairs
    assert 5 not in cf.cycle_nodes, "Node 5 (altruist) should not be in the list of cycle nodes."
    assert 0 in cf.cycle_nodes, "Node 0 should be a cycle node."

def test_find_cycles(cf_setup):
    """
    Verify if it finds the expected cycles in the graph.
      - Cycle A: 0 -> 1 -> 4 -> 0 (Size 3)
      - Cycle B: 0 -> 2 -> 3 -> 0 (Size 3)
    """
    g, characterization = cf_setup
    cf = CycleFinder(g, characterization)
    cf.find_cycles(max_cycle=3)

    assert len(cf.cycles) == 2, "Should find exactly 2 cycles of size <= 3."

    # Normalize sets to verify presence regardless of order
    found_cycles_sets = [set(c) for c in cf.cycles]

    assert {0, 1, 4} in found_cycles_sets, "The cycle {0, 1, 4} should have been found."
    assert {0, 2, 3} in found_cycles_sets, "The cycle {0, 2, 3} should have been found."

def test_find_chains(cf_setup):
    """
    Verify the chains starting from the altruist nodes (only node 5).
    """
    g, characterization = cf_setup
    cf = CycleFinder(g, characterization)
    cf.find_chains(max_chain=3)

    expected_chains = [
        [5, 3],
        [5, 3, 0],
        [5, 6],
        [5, 6, 1]
    ]

    for expected in expected_chains:
        assert expected in cf.chains, f"The chain {expected} should have been found."

    assert len(cf.chains) == 4, "Should find exactly 4 chains."

def test_inverted_graph_initialization(cf_setup):
    """
    Covers lines 24-25: Initialization when graph is inverted.
    """
    g, characterization = cf_setup

    # Invert the graph manually
    g.reverse() # This sets g.inverted = True

    cf = CycleFinder(g, characterization)

    # When inverted, lists should be sorted in descending order
    # Helper to check if list is sorted descending
    def is_sorted_descending(lst):
        return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))

    assert is_sorted_descending(cf.cycle_nodes), "Cycle nodes should be sorted descending for inverted graphs"
    assert is_sorted_descending(cf.altruistic_nodes), "Altruistic nodes should be sorted descending for inverted graphs"

def test_self_loop_cycle():
    """
    Test Self-loops (cycles of size 1). 
    Based on our generator implementation, self-loops will not exist but we test the logic 
    if in the future "Compatible" Pairs appear in the graph file with self-loops.
    """
    # Create graph with a self-loop (0->0)
    g = Digraph()
    g.nv = 1
    g.verts = [0]
    g.adjList = [[0]]
    g.ne = 1

    # Mock characterization
    characterization = {
        0: {
            COL_PAIR_ID: 0,
            COL_DONOR_ID: 0,
            COL_PATIENT_ID: 0,
            COL_PAIR_DESCRIPTION: "pair"
        }
    }

    cf = CycleFinder(g, characterization)
    cf.find_cycles(max_cycle=3)

    assert [0] in cf.cycles, "Should detect self-loop (cycle of size 1)"
    assert len(cf.cycles) == 1, "Should find exactly 1 cycle (the self-loop)"
