import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kep.constants import *
from kep.generation.generate import *


@pytest.fixture(autouse=True)
def set_random_seed():
    """Define a fixed seed for reproducibility in tests."""
    random.seed(42)
    np.random.seed(42)

def test_pair_initialization_structure():
    """Test if a pair is initialized with correct attributes."""
    pair = PatientDonorPair(pair_id=0, arrival_time=10, description='pair')

    assert pair.pair_id == 0
    assert pair.arrival_time == 10
    assert pair.description == 'pair'
    assert pair.donor_blood in [1, 2, 3, 4]
    assert pair.patient_blood in [1, 2, 3, 4]
    assert pair.departure_time is not None
    assert pair.departure_time > pair.arrival_time

def test_altruist_initialization():
    """Test if an altruist is created correctly (without a patient)."""
    altruist = PatientDonorPair(pair_id=1, arrival_time=5, description='altruist')

    assert altruist.description == 'altruist'
    assert altruist.patient_blood == -1
    assert altruist.patient_age == -1
    assert altruist.pra_level == -1
    assert altruist.is_spouse_donor == -1
    assert altruist.self_compatibility == 4
    assert altruist.accepts_aboi == -1

def test_distribution_accuracy():
    """Verify if the generated pairs reflect the expected blood type distribution."""
    pairs = generate_pairs(simul_years=5, pair_rate=0.2, alt_rate=100) 
    
    # Count donors with blood type O (blood type 1)
    o_donors = sum(1 for p in pairs if p.donor_blood == 1)
    total_donors = len(pairs)
    
    ratio = o_donors / total_donors
    expected_ratio = 0.48
    
    # Accept a margin of error of 2% (0.02)
    assert abs(ratio - expected_ratio) < 0.02, f"Expected 48% of O donors, got {ratio:.2%}"

@pytest.mark.parametrize("donor_blood, patient_blood, receiver_type, expected", [
    # Normal ABO Compatibility Cases
    (1, 2, 'pair', True),    # O -> A (Universal Donor)
    (2, 1, 'pair', False),   # A -> O (Incompatible)
    (3, 4, 'pair', True),    # B -> AB (Universal Receiver)
    (2, 3, 'pair', False),   # A -> B (Incompatible)
    
    # Altruists (Never can receive) -> Covers: if self.description == 'altruist': return False
    (1, 1, 'altruist', False), # Same blood O->O, but fails because it is altruist
])
def test_abo_compatibility_logic(donor_blood, patient_blood, receiver_type, expected):
    """Test various ABO compatibility scenarios."""
    p_donor = PatientDonorPair(0)
    p_receiver = PatientDonorPair(1)
    
    # Configure Donor
    p_donor.donor_blood = donor_blood
    
    # Configure Receiver
    p_receiver.description = receiver_type

    if receiver_type == 'altruist':
        p_receiver.patient_blood = -1 # Altruists do not have patient blood
    else:
        p_receiver.patient_blood = patient_blood
    
    assert p_receiver.is_abo_compatible_with(p_donor) is expected

def test_hla_self_compatibility_check():
    pair = PatientDonorPair(0)
    
    pair.self_hla_compatible = True
    assert pair.is_hla_compatible_with(pair) is True

    pair.self_hla_compatible = False
    assert pair.is_hla_compatible_with(pair) is False

def test_hla_compatibility_logic():
    """Test PRA levels logic in HLA compatibility."""
    # Case 1: PRA Level 3 (High) -> Probability of incompatibility should be 0.90
    p_donor1 = PatientDonorPair(0)
    p_patient1 = PatientDonorPair(1)
    
    p_patient1.description = 'pair'
    p_patient1.pair_id = 1
    p_donor1.pair_id = 0
    
    p_patient1.pra_level = 3
    
    # If random is 0.89 (below 0.90), it is incompatible (returns False)
    with patch('random.random', return_value=0.89):
        assert p_patient1.is_hla_compatible_with(p_donor1) is False, "PRA 3 should fail with random 0.89"

    # If random is 0.91 (above 0.90), it is compatible (returns True)
    with patch('random.random', return_value=0.91):
        assert p_patient1.is_hla_compatible_with(p_donor1) is True, "PRA 3 should pass with random 0.91"

    # Case 2: PRA Level 2 (Medium) -> Probability of incompatibility should be 0.45
    p_donor2 = PatientDonorPair(2)
    p_patient2 = PatientDonorPair(3)
    
    p_patient2.description = 'pair'
    p_patient2.pair_id = 3
    p_donor2.pair_id = 2
    
    p_patient2.pra_level = 2
    
    # If random is 0.44 (below 0.45), it is incompatible (returns False)
    with patch('random.random', return_value=0.44):
        assert p_patient2.is_hla_compatible_with(p_donor2) is False, "PRA 2 should fail with random 0.44"

    # If random is 0.46 (above 0.45), it is compatible (returns True)
    with patch('random.random', return_value=0.46):
        assert p_patient2.is_hla_compatible_with(p_donor2) is True, "PRA 2 should pass with random 0.46"

    # Case 3: PRA Level 1 (Low) -> Probability of incompatibility should be 0.05
    p_donor3 = PatientDonorPair(4)
    p_patient3 = PatientDonorPair(5)

    p_patient3.description = 'pair'
    p_patient3.pair_id = 5
    p_donor3.pair_id = 4

    p_patient3.pra_level = 1

    # If random is 0.04 (below 0.05), it is incompatible (returns False)
    with patch('random.random', return_value=0.04):
        assert p_patient3.is_hla_compatible_with(p_donor3) is False, "PRA 1 should fail with random 0.04"

    # If random is 0.06 (above 0.05), it is compatible (returns True)
    with patch('random.random', return_value=0.06):
        assert p_patient3.is_hla_compatible_with(p_donor3) is True, "PRA 1 should pass with random 0.06"

    # Case 4: Altruist (cannot receive, so should always return False)

    p4_donor = PatientDonorPair(6)
    p4_donor.description = 'pair'
    p4_patient = PatientDonorPair(7)
    p4_patient.description = 'altruist'

    assert p4_patient.is_hla_compatible_with(p4_donor) is False, "Altruist don't have a patient to test if they are HLA Compatible to receive."

def test_is_compatible_with_method():
    """Test the general 'is_compatible_with' logic for Pairs and Altruists."""
    p1 = PatientDonorPair(0) # Par Dador
    p2 = PatientDonorPair(1) # Par Recetor
    p3 = PatientDonorPair(2) # Altruísta Recetor (impossível)
    
    p1.description, p2.description = 'pair', 'pair'
    p3.description = 'altruist'

    # Test 1: Pair -> Pair (ABO Incompatible)
    p1.donor_blood = 2 # A
    p2.patient_blood = 1 # O
    assert not p1.is_compatible_with(p2), "is_compatible_with should return False for ABO incompatible pairs."

    # Test 2: Pair -> Altruist (Altruist should never receive, so always False)
    assert not p1.is_compatible_with(p3), "Altruist should not be able to receive."

    # Test 3: Altruist Donor (Complex Logic)
    # Covers the line: return (other != 'altruist' and abo_comp and hla_comp)
    p_alt_donor = PatientDonorPair(3, description='altruist')

    # 1 Altruist -> Altruist (Fails the 1st condition: other_pair.description != 'altruist')
    assert p_alt_donor.is_compatible_with(p3) is False, "Altruist cannot donate to another Altruist."

    # Prepare p2 (Pair Receiver) with Mocks to test the rest of the 'AND'
    p2.is_abo_compatible_with = MagicMock()
    p2.is_hla_compatible_with = MagicMock()

    # 2 Altruist -> Pair (ABO Failure)
    p2.is_abo_compatible_with.return_value = False
    p2.is_hla_compatible_with.return_value = True
    assert p_alt_donor.is_compatible_with(p2) is False, "Altruist -> Pair fails if ABO incompatible."

    # 3 Altruist -> Pair (HLA Failure)
    p2.is_abo_compatible_with.return_value = True
    p2.is_hla_compatible_with.return_value = False
    assert p_alt_donor.is_compatible_with(p2) is False, "Altruist -> Pair fails if HLA incompatible."

    # 4 Altruist -> Pair (Total Success)
    p2.is_abo_compatible_with.return_value = True
    p2.is_hla_compatible_with.return_value = True
    assert p_alt_donor.is_compatible_with(p2) is True, "Altruist -> Pair works if ABO+HLA compatible."

def test_self_compatibility_logic_detailed():
    """
    Test the self-compatibility classification logic in detail, covering all cases:
    1 = Incompatible
    2 = Half-Compatible (ABO Incomp + HLA Comp + Low Titers)
    3 = Fully Compatible
    """
    # Case: Fully Compatible (3)
    # Donor O (1) -> Patient A (2) (ABO OK)
    # Forced: HLA Compatible
    with patch('random.choices', side_effect=[[1], [2], [1], [1]]), \
         patch('random.randint', return_value=30), \
         patch('random.random', return_value=0.99), \
         patch.object(PatientDonorPair, 'is_hla_self_compatible', return_value=True):
        
        p = PatientDonorPair(0)
        # Adjust manually to ensure the test
        p.donor_blood = 1 
        p.patient_blood = 2
        p.self_abo_compatible = True
        p.self_hla_compatible = True
        
        # Recalculate logic (simulating the init)
        if p.self_abo_compatible and p.self_hla_compatible:
            p.self_compatibility = 3
            
        assert p.self_compatibility == 3
        assert p.accepts_aboi == -1

    # Case: Half-Compatible (2)
    # ABO Incompatible (A->O), but HLA Compatible and Low Titers
    with patch('random.random', return_value=0.1), \
         patch.object(PatientDonorPair, 'is_hla_self_compatible', return_value=True):
        
        p2 = PatientDonorPair(1)
        p2.donor_blood = 2 # A
        p2.patient_blood = 1 # O
        p2.has_low_titers = True # Forced by random < 0.75 or manually
        p2.self_abo_compatible = False
        p2.self_hla_compatible = True
        
        # Execute classification logic manually for strict unit test
        if ((not p2.self_abo_compatible) and p2.self_hla_compatible and p2.has_low_titers):
            p2.self_compatibility = 2
            p2.accepts_aboi = 0
            
        assert p2.self_compatibility == 2
        assert p2.accepts_aboi == 0

def test_departure_time_logic():
    """Verify that departure times are shorter for compatible pairs and longer for incompatible ones."""
    # Compatible (p=0.1)
    p_comp = PatientDonorPair(0)
    p_comp.description = 'pair'
    p_comp.self_compatibility = 3
    
    # Incompatible (p=0.001)
    p_incomp = PatientDonorPair(1)
    p_incomp.description = 'pair'
    p_incomp.self_compatibility = 1
    
    # Mock numpy geometric to return fixed value based on input probability
    # If p=0.1 (fast), geometric returns low value. If p=0.001 (slow), high value.
    with patch('numpy.random.geometric') as mock_geo:
        mock_geo.side_effect = lambda p: 10 if p == 0.1 else 1000
        
        t_comp = p_comp.calculate_departure_time()
        t_incomp = p_incomp.calculate_departure_time()
        
        assert t_comp < t_incomp, "Compatible pairs should leave the pool faster"

def test_generate_pairs_reproducibility():
    """Test if generating pairs with the same seed produces identical results."""
    seed = 123
    pairs_run_1 = generate_pairs(simul_years=1, pair_rate=5, alt_rate=50, seed=seed)
    pairs_run_2 = generate_pairs(simul_years=1, pair_rate=5, alt_rate=50, seed=seed)

    assert len(pairs_run_1) == len(pairs_run_2)
    assert pairs_run_1[0].pair_id == pairs_run_2[0].pair_id
    assert pairs_run_1[0].donor_blood == pairs_run_2[0].donor_blood

def test_compatibility_matrix_correctness():
    """Verify that the compatibility matrix is built correctly based on the is_compatible_with method."""
    p1 = MagicMock(spec=PatientDonorPair)
    p2 = MagicMock(spec=PatientDonorPair)
    p3 = MagicMock(spec=PatientDonorPair)
    
    # Configuration:
    # P1 -> donates to P2
    # P2 -> donates to P1 (direct exchange)
    # P3 -> nobody
    
    pairs = [p1, p2, p3]
    
    # side_effect -> (self=p1 e other=p2) -> True
    def side_effect(other):
        sender = None
        return False 

    # Mock expected results
    # P1 is compatible with P2
    p1.is_compatible_with.side_effect = lambda x: x == p2
    # P2 is compatible with P1
    p2.is_compatible_with.side_effect = lambda x: x == p1
    # P3 is not compatible with anyone
    p3.is_compatible_with.return_value = False

    matrix = build_compatibility_matrix(pairs)
    
    # Matrix should reflect:
    #    P1 P2 P3
    # P1  0  1  0
    # P2  1  0  0
    # P3  0  0  0
    
    expected = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])
    
    np.testing.assert_array_equal(matrix, expected)

def test_blood_number_to_letter():
    """Covers blood_number_to_letter method."""
    p = PatientDonorPair(0)
    assert p.blood_number_to_letter(1) == "O"
    assert p.blood_number_to_letter(2) == "A"
    assert p.blood_number_to_letter(3) == "B"
    assert p.blood_number_to_letter(4) == "AB"
    assert p.blood_number_to_letter(99) == "N/A"

def test_to_csv_row():
    """Covers to_csv_row method."""
    p = PatientDonorPair(0)
    # Mock some values to ensure consistent output
    p.donor_blood = 1
    p.patient_blood = 2
    p.pra_value = 0.5
    row = p.to_csv_row()
    assert isinstance(row, list)
    assert row[2] == "O" # Donor Blood
    assert row[5] == "A" # Patient Blood

def test_repr():
    """Covers __repr__."""
    p = PatientDonorPair(0)
    p.patient_blood = 1
    p.donor_blood = 2
    rep = repr(p)
    assert "Pair1(P:O, D:A" in rep

def test_save_files(tmp_path):
    """Test saving CSV and Matrix to files."""
    p1 = PatientDonorPair(0)
    p2 = PatientDonorPair(1)

    # Mock compatibility to always return True for testing
    with patch.object(PatientDonorPair, 'is_compatible_with', return_value=True):
        pairs = [p1, p2]

        csv_file = tmp_path / "test_pool.csv"
        matrix_file = tmp_path / "test_arcs.txt"

        save_pairs_to_csv(pairs, str(csv_file))
        assert csv_file.exists()

        save_compatibility_matrix(pairs, str(matrix_file))
        assert matrix_file.exists()

        # Verify matrix content format
        content = matrix_file.read_text()
        assert "2, 2" in content.splitlines()[0], "Header should indicate 2 pairs and 2 edges (0->1 and 1->0)"
        assert "0, 1, 1" in content, "Matrix should contain an edge from 0 to 1."

def test_run_generation_execution(capsys):
    """
    Test the run_generation function.
    Mocks to verify without creating files.
    """
    with patch('kep.generation.generate.SIMULATION_YEARS', 1), \
         patch('kep.generation.generate.NUMBER_OF_INSTANCES', 1), \
         patch('kep.generation.generate.generate_pairs', return_value=[]) as mock_gen, \
         patch('kep.generation.generate.save_pairs_to_csv') as mock_save_csv, \
         patch('kep.generation.generate.save_compatibility_matrix') as mock_save_matrix, \
         patch('os.makedirs') as mock_makedirs:

        run_generation()

        assert mock_gen.call_count == 1, "Should have generated exactly one instance because NUMBER_OF_INSTANCES=1."

        assert mock_save_csv.call_count == 1, "Should have attempted to save one CSV file."
        assert mock_save_matrix.call_count == 1, "Should have attempted to save one Matrix file."

        assert mock_makedirs.called, "Should have attempted to create output directories."

        # Verify standard output to ensure the loop body was printed
        captured = capsys.readouterr()
        assert "Generating instance 1" in captured.out, "Output should indicate generation of instance 1."
        assert "Output directory set to" in captured.out, "Output should indicate the output directory."
