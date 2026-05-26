import os

import pytest

from kep.compatibility_graph import digraph
from kep.constants import *


def read_char_for_test(filepath):
    """Read characterization file for testing."""
    char_dict = {}
    with open(filepath, "r") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line: continue

            line = [el.strip() for el in line.split(",")]
            if len(line) <= COL_PAIR_COMPATIBILITY: continue

            pair_id = int(line[COL_PAIR_ID])
            char_dict[pair_id] = {
                COL_PAIR_ID: pair_id,
                COL_DONOR_ID: int(line[COL_DONOR_ID]),
                COL_DONOR_BLOOD: str(line[COL_DONOR_BLOOD]),
                COL_DONOR_AGE: int(line[COL_DONOR_AGE]),
                COL_PATIENT_ID: int(line[COL_PATIENT_ID]),
                COL_PATIENT_BLOOD: str(line[COL_PATIENT_BLOOD]),
                COL_PATIENT_PRA: float(line[COL_PATIENT_PRA]),
                COL_PATIENT_ACCEPT_IMSUP: str(line[COL_PATIENT_ACCEPT_IMSUP]).lower() in ("true", "1"),
                COL_PATIENT_AGE: int(line[COL_PATIENT_AGE]),
                COL_PAIR_ARRIVAL: float(line[COL_PAIR_ARRIVAL]),
                COL_PAIR_DEPARTURE: float(line[COL_PAIR_DEPARTURE]),
                COL_PAIR_DESCRIPTION: str(line[COL_PAIR_DESCRIPTION]).strip().lower(),
                COL_PAIR_COMPATIBILITY: int(line[COL_PAIR_COMPATIBILITY])
            }
    return char_dict

@pytest.fixture
def load_kep_data():
    """
    Returns a FUNCTION that loads specific files.
    Usage: data = load_kep_data("file.csv", "file.txt")
    """
    def _loader(csv_filename, txt_filename):
        base_path = os.path.dirname(__file__)
        data_dir = os.path.join(base_path, "tests_data")

        char_path = os.path.join(data_dir, csv_filename)
        graph_path = os.path.join(data_dir, txt_filename)

        if not os.path.exists(graph_path) or not os.path.exists(char_path):
            pytest.fail(f"Files not found: {csv_filename} or {txt_filename}")

        # Load Graph
        G = digraph.Digraph()
        G.load(graph_path)

        characterization = read_char_for_test(char_path)

        data = {
            "characterization": characterization,
            "epoch": 1,
            "tbm": 90.0
        }

        return G, data

    return _loader
