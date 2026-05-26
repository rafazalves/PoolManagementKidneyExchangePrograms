import csv
import os
import random
from math import log

import numpy as np

from kep.constants import *


class PatientDonorPair:
    """Represents a patient-donor pair in the kidney exchange pool"""

    def __init__(self, pair_id, arrival_time=0, description='pair'):
        self.pair_id = pair_id
        self.arrival_time = arrival_time
        self.description = description  # pair or altruist

        # Blood groups: 1=O (48%), 2=A (34%), 3=B (14%), 4=AB (4%) Data from Roth et al. (2007)
        # For altruistic donors, patient fields are left as None.
        self.donor_blood = random.choices([1, 2, 3, 4],
                                         weights=[0.48, 0.34, 0.14, 0.04])[0]

        self.donor_age = random.randint(18, 73)

        if description == 'altruist':
            self.patient_blood = -1
            self.patient_age = -1
        else:
            self.patient_blood = random.choices([1, 2, 3, 4],
                                               weights=[0.48, 0.34, 0.14, 0.04])[0]
            self.patient_age = random.randint(18, 73)

        # PRA levels: 1=low (70%), 2=medium (20%), 3=high (10%) Data from Roth et al. (2007)
        if description == 'altruist':
            self.pra_level = -1
        else:
            self.pra_level = random.choices([1, 2, 3],
                                           weights=[0.7, 0.2, 0.1])[0]

        # Spouse donor: 1=non-spouse (90%), 2=spouse (10%) Data from Andersson and Kratz (2019)
        if description == 'altruist':
            self.is_spouse_donor = -1
        else:
            self.is_spouse_donor = random.choices([1, 2],
                                                 weights=[0.9, 0.1])[0]

        # Generate actual PRA value based on level. Data from Roth et al. (2007)
        if description == 'altruist':
            self.pra_value = -1
        else:
            if self.pra_level == 1:
                self.pra_value = random.uniform(0, 0.1)
            elif self.pra_level == 2:
                self.pra_value = random.uniform(0.1, 0.8)
            else:
                self.pra_value = random.uniform(0.8, 1.0)

        # ABOi compatibility (75% feasible across blood barrier)
        self.has_low_titers = random.random() < 0.75

        if self.description != 'altruist':
            self.self_abo_compatible = self.is_abo_self_compatible()
            self.self_hla_compatible = self.is_hla_self_compatible()
            """
                Check if patient can receive from their own donor
                Compatibility: 
                - 'fully_compatible': 3
                - 'half_compatible': 2
                - 'incompatible': 1
            """
            # Fully compatible: ABO + HLA compatible
            if self.self_abo_compatible and self.self_hla_compatible:
                self.self_compatibility = 3
            # Half-compatible: ABO incompatible but HLA compatible and low titers
            elif ((not self.self_abo_compatible) and self.self_hla_compatible and self.has_low_titers):
                self.self_compatibility = 2
            # Incompatible: all other cases
            else:
                self.self_compatibility = 1
        else:
            self.self_compatibility = 4
            self.self_abo_compatible = False
            self.self_hla_compatible = False

        # How many match runs / epochs the patient is willing to wait until accepting ABOi transplant
        if self.self_compatibility == 2:
            self.accepts_aboi = 0 # 100% accepts immediately (run "generate_acceptance_variations.py" to create variations with different probabilities)
        else:
            self.accepts_aboi = -1

        # Calculate departure time
        self.departure_time = self.calculate_departure_time()


    def is_abo_self_compatible(self):
        # O donor is universal
        if self.donor_blood == 1:
            return True
        # AB patient is universal receiver
        if self.patient_blood == 4:
            return True
        # Same blood type is compatible
        return self.patient_blood == self.donor_blood

    def is_abo_compatible_with(self, other_pair):
        """Check if this pair's patient is ABO compatible with other pair's donor"""
        # If this is an altruist (no patient) then they cannot receive
        if self.description == 'altruist':
            return False
        # O donor is universal
        if other_pair.donor_blood == 1:
            return True
        # AB patient is universal receiver
        if self.patient_blood == 4:
            return True
        # Same blood type is compatible
        return self.patient_blood == other_pair.donor_blood

    def is_hla_self_compatible(self):
        """Draw HLA compatibility with own donor"""
        spouse_flag = self.is_spouse_donor

        # Determine probability of positive crossmatch (incompatible)
        if self.pra_level == 1 and spouse_flag == 1:
            prob_positive = 0.05
        elif self.pra_level == 2 and spouse_flag == 1:
            prob_positive = 0.45
        elif self.pra_level == 3 and spouse_flag == 1:
            prob_positive = 0.90
        elif self.pra_level == 1 and spouse_flag == 2:
            prob_positive = 0.2875
        elif self.pra_level == 2 and spouse_flag == 2:
            prob_positive = 0.5875
        else: # self.pra_level == 3 and spouse_flag == 2
            prob_positive = 0.9225

        # Return True if negative crossmatch (i.e. compatible)
        return random.random() > prob_positive

    def is_hla_compatible_with(self, other_pair):
        """Check HLA compatibility (crossmatch test) between this pair's patient and other pair's donor."""
        # If this is an altruist (no patient) then they cannot receive
        if self.description == 'altruist':
            return False

        if self.pair_id == other_pair.pair_id:
            return self.self_hla_compatible

        # Donor from another pair → treat as non-spouse (spouse_flag = 1), so use only PRA level
        # Determine probability of positive crossmatch (incompatible)
        if self.pra_level == 1:
            prob_positive = 0.05
        elif self.pra_level == 2:
            prob_positive = 0.45
        else:
            prob_positive = 0.90

        # Return True if negative crossmatch (i.e. compatible)
        return random.random() > prob_positive

    def is_compatible_with(self, other_pair):
        """
        Returns True if this pair's donor can donate to the other pair's patient.
        """
        # Altruists can only donate
        if self.description == 'altruist':
            # other must have a patient
            return (other_pair.description != 'altruist' and other_pair.is_abo_compatible_with(self) and other_pair.is_hla_compatible_with(self))

        # Altruists cannot receive
        if other_pair.description == 'altruist':
            return False

        return (other_pair.is_abo_compatible_with(self) and other_pair.is_hla_compatible_with(self))

    def calculate_departure_time(self):
        """Calculate departure time based on pair type and probability of departure"""
        if self.description == 'altruist':
            p_departure = 0.01
        elif self.self_compatibility == 3:  # Compatible
            p_departure = 0.1
        else:
            # Incompatible (1) e Half-Compatible (2)
            p_departure = 0.001

        # We add 89 days to ensure minimum stay of 90 days (1 matching period)
        duration = 89 + np.random.geometric(p_departure)

        return self.arrival_time + duration


    def blood_number_to_letter(self, blood_number):
        """Convert blood group number to letter representation"""
        return {1: "O", 2: "A", 3: "B", 4: "AB"}.get(blood_number, "N/A")

    def to_csv_row(self):
        """Convert pair data to CSV row format"""
        return [
            self.pair_id,  # COL_PAIR_ID
            self.pair_id,  # COL_DONOR_ID
            self.blood_number_to_letter(self.donor_blood),  # COL_DONOR_BLOOD
            self.donor_age,  # COL_DONOR_AGE
            self.pair_id,  # COL_PATIENT_ID
            self.blood_number_to_letter(self.patient_blood),  # COL_PATIENT_BLOOD
            round(self.pra_value, 3),  # COL_PATIENT_PRA
            self.accepts_aboi,  # COL_PATIENT_ACCEPT_IMSUP
            self.patient_age,  # COL_PATIENT_AGE
            round(self.arrival_time, 2),  # COL_PAIR_ARRIVAL
            round(self.departure_time, 2),  # COL_PAIR_DEPARTURE
            self.description.upper(),  # COL_PAIR_DESCRIPTION
            self.self_compatibility  # COL_PAIR_COMPATIBILITY
        ]

    def __repr__(self):
        return (f"Pair{self.pair_id+1}(P:{self.blood_number_to_letter(self.patient_blood)}, "
                f"D:{self.blood_number_to_letter(self.donor_blood)}, PRA:{self.pra_level}, Compatibility:{self.self_compatibility})")

def exponential(rate):
    """Exponential distribution to model time between arrivals."""
    return -log(random.random()) / rate

def generate_arrival_times(maxtime, rate):
    """Generates a list of arrival times between 0 and maxtime according to a given rate."""
    arrival_times = []
    ctime = exponential(rate)

    while ctime < maxtime:
        arrival_times.append(ctime)
        ctime += exponential(rate)

    return arrival_times

def generate_pairs(simul_years=12, pair_rate=1/3.6, alt_rate=1/75.0, seed=None):
    """
    Generate patient-donor pairs with exponential arrival times.
    
    Parameters:
    - simul_years: Number of years to simulate
    - pair_rate: Arrival rate in days for pairs
    - alt_rate: Arrival rate in days for altruistic donors
    - seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Convert years to days (360 days per year (1 year = 12 months x 30 days))
    simul_time = simul_years * 360

    # Generate arrival times for each type
    pair_arrival_times = generate_arrival_times(simul_time, pair_rate)
    alt_arrival_times = generate_arrival_times(simul_time, alt_rate)

    print(f"Generated {len(pair_arrival_times)} pairs")
    print(f"Generated {len(alt_arrival_times)} altruistic donors")
    print(f"Total pairs: {len(pair_arrival_times) + len(alt_arrival_times)}")
    print()

    pairs = []
    pair_id = 0

    # Generate patient-donor pairs
    for arrival_time in pair_arrival_times:
        pair = PatientDonorPair(pair_id=pair_id, arrival_time=arrival_time, description='pair')

        # To adjust distribution, only keep 10% of fully compatible pairs
        if pair.self_compatibility == 3 and random.random() > 0.10:
            continue

        pairs.append(pair)
        pair_id += 1

    # Generate altruistic donors
    for arrival_time in alt_arrival_times:
        pair = PatientDonorPair(pair_id=pair_id, arrival_time=arrival_time, description='altruist')
        pairs.append(pair)
        pair_id += 1

    return pairs


def build_compatibility_matrix(pairs):
    """Build compatibility matrix for a list of pairs"""
    n = len(pairs)
    compatibility = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i != j:  # Can't exchange with yourself
                if pairs[i].is_compatible_with(pairs[j]):
                    compatibility[i][j] = 1

    return compatibility

def save_pairs_to_csv(pairs, filepath):
    """Save all pairs data to a CSV file"""
    # Define column headers
    headers = [
        "COL_PAIR_ID",
        "COL_DONOR_ID",
        "COL_DONOR_BLOOD",
        "COL_DONOR_AGE",
        "COL_PATIENT_ID",
        "COL_PATIENT_BLOOD",
        "COL_PATIENT_PRA",
        "COL_PATIENT_ACCEPT_IMSUP",
        "COL_PATIENT_AGE",
        "COL_PAIR_ARRIVAL",
        "COL_PAIR_DEPARTURE",
        "COL_PAIR_DESCRIPTION",
        "COL_PAIR_COMPATIBILITY"
    ]

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    # Write CSV
    with open(filepath, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for pair in pairs:
            writer.writerow(pair.to_csv_row())

    print(f"Data saved to '{filepath}' successfully.")

def save_compatibility_matrix(pairs, filepath):
    """Save compatibility matrix"""
    n = len(pairs)
    compatibility_matrix = build_compatibility_matrix(pairs)
    total_compatibilities = np.sum(compatibility_matrix)

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    with open(filepath, 'w') as f:
        # Write first line: <number of pairs> <number of arcs>
        f.write(f"{n}, {total_compatibilities}\n")

        # Write each compatibility as: donor_id, patient_id, 1
        for i in range(n):
            for j in range(n):
                if compatibility_matrix[i][j] == 1:
                    f.write(f"{i}, {j}, 1\n")

    print(f"Compatibility matrix saved to '{filepath}' successfully.")


def run_generation():
    """
    Main function to run the data generation process.
    """
    # Years to simulate
    simul_years = SIMULATION_YEARS

    # Generate multiple instances with different seeds
    n_instances = NUMBER_OF_INSTANCES

    # Arrival rates (in days)
    pair_rate = 1/3.6     # All pairs
    alt_rate = 1/75       # Altruistic donors

    # Create output directory path
    base_folder = "data"
    sub_folder_pool = f"{simul_years}_year_simulation/100_percentage/"
    sub_folder_arcs = f"{simul_years}_year_simulation/"
    output_directory_pool = os.path.join(base_folder, sub_folder_pool)
    output_directory_arcs = os.path.join(base_folder, sub_folder_arcs)

    os.makedirs(output_directory_pool, exist_ok=True)
    print(f"Output directory set to: {output_directory_pool}")

    for instance_id in range(1, n_instances + 1):
        seed = instance_id
        print(f"\n{'='*60}")
        print(f"Generating instance {instance_id} (seed={seed})")
        print(f"{'='*60}\n")

        pairs = generate_pairs(
            simul_years=simul_years,
            pair_rate=pair_rate,
            alt_rate=alt_rate,
            seed=seed
        )

        # Save generated data
        csv_filename = f"pool_{instance_id}.csv"
        arcs_filename = f"arcs_{instance_id}.txt"

        csv_path = os.path.join(output_directory_pool, csv_filename)
        arcs_path = os.path.join(output_directory_arcs, arcs_filename)

        save_pairs_to_csv(pairs, csv_path)
        save_compatibility_matrix(pairs, arcs_path)
