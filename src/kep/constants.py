# Characterization file format (columns)
COL_PAIR_ID = 0  # A
COL_DONOR_ID = 1  # B
COL_DONOR_BLOOD = 2  # C - Possible values: O, A, B, AB
COL_DONOR_AGE = 3  # D
COL_PATIENT_ID = 4  # E
COL_PATIENT_BLOOD = 5  # F - Possible values: O, A, B, AB
COL_PATIENT_PRA = 6  # G
COL_PATIENT_ACCEPT_IMSUP = 7  # H - Number of epochs/match runs patient is willing to wait until accepts immunosuppressants
COL_PATIENT_AGE = 8  # I
COL_PAIR_ARRIVAL = 9  # J
COL_PAIR_DEPARTURE = 10  # K
COL_PAIR_DESCRIPTION = 11  # L - Possible values: 'pair', 'altruist'
COL_PAIR_COMPATIBILITY = 12  # M - Possible values: 1 (incompatible), 2 (half_compatible), 3 (fully_compatible)

# Simulation constants
SIMULATION_YEARS = 24 # Number of years to simulate (6, 12, 24)
NUMBER_OF_INSTANCES = 100
MAX_CYCLE_LENGTH = 3
MAX_CHAIN_LENGTH = 3
MATCHING_PERIOD_DAYS = 90
ACCEPTANCE_IMMUNOSUPPRESSION_PERCENTAGE = 100 # Percentage of patients accepting immunosuppressants (0, 25, 50, 75, 100)

# Weights for types of compatibility prioritization
HIERARCHY_MULTIPLIER = 10000000
INCOMPATIBLE_BONUS = 0.01
HALFCOMPATIBLE_BONUS = 0.001

# Control for printing logs in simulator
LOG = False
