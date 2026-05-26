import csv

from kep.constants import *


def read_characterization(path):
    charact = []

    with open(path, "r") as csvfile:
        header = csvfile.readline()  # Skip the header line

        cr = csv.reader(csvfile)

        for row in cr:
            # Skip empty rows
            if not row or all(cell.strip() == '' for cell in row):
                print(f"Warning: Skipping empty row: {row}")
                continue
            # Ensure row has enough columns (at least up to index 12)
            if len(row) < 13:
                print(f"Warning: Skipping malformed row: {row}")
                continue
            update_row(row)
            charact.append(row)

    return charact

def update_row(row):
    row[COL_PAIR_ID] = int(row[COL_PAIR_ID])
    row[COL_DONOR_ID] = int(row[COL_DONOR_ID])
    row[COL_DONOR_AGE] = int(row[COL_DONOR_AGE])
    row[COL_PATIENT_ID] = int(row[COL_PATIENT_ID])
    row[COL_PATIENT_PRA] = float(row[COL_PATIENT_PRA])
    row[COL_PATIENT_ACCEPT_IMSUP] = int(row[COL_PATIENT_ACCEPT_IMSUP])
    row[COL_PATIENT_AGE] = int(row[COL_PATIENT_AGE])
    row[COL_PAIR_ARRIVAL] = float(row[COL_PAIR_ARRIVAL])
    row[COL_PAIR_DEPARTURE] = float(row[COL_PAIR_DEPARTURE])
    row[COL_PAIR_DESCRIPTION] = str(row[COL_PAIR_DESCRIPTION].replace(" ", "")).lower()
    row[COL_DONOR_BLOOD] = str(row[COL_DONOR_BLOOD].replace(" ", ""))
    row[COL_PATIENT_BLOOD] = str(row[COL_PATIENT_BLOOD].replace(" ", ""))
    row[COL_PAIR_COMPATIBILITY] = int(row[COL_PAIR_COMPATIBILITY])

def read_graph(path):
    verts = set()
    arcs = []
    with open(path, 'r') as f:
        f.readline() # skip header count
        for line in f:
            s, d, w = line.strip().split(',')
            s, d = int(s), int(d)
            verts.add(s)
            verts.add(d)
            arcs.append((s, d))
    return list(verts), arcs

def read_times(charact):
    # Returns dict: id -> (arrival, departure), min_time, max_time
    times = {}
    mn, mx = 999999.0, -1.0
    for row in charact:
        pid = int(row[COL_PAIR_ID])
        arr = float(row[COL_PAIR_ARRIVAL])
        dep = float(row[COL_PAIR_DEPARTURE])
        times[pid] = (arr, dep)
        if arr < mn: mn = arr
        if dep > mx: mx = dep
    return times, mn, mx

def log_pair_history(pair_id, departure_time, status, times, characterization, file_handle, epoch, match_type="NA", match_group="NA"):
    """
    Write into the history file the detailed data of the pair.
    """
    arrival = times[pair_id][0]
    wait = departure_time - arrival

    p_data = characterization[pair_id]

    try:
        blood_p = p_data[COL_PATIENT_BLOOD]
        blood_d = p_data[COL_DONOR_BLOOD]
        raw_pra = float(p_data[COL_PATIENT_PRA])
        description = p_data[COL_PAIR_DESCRIPTION]

        if description == 'altruist':
            pra_category = "NA"
        else:
            if raw_pra <= 0.1:
                pra_category = "Low"
            elif raw_pra <= 0.8:
                pra_category = "Medium"
            else:
                pra_category = "High"

        comp_type = p_data[COL_PAIR_COMPATIBILITY] # 1=Incompat, 2=Half, 3=Compat

    except IndexError:
        blood_p, blood_d, pra_category, comp_type, description = ("ERR", "ERR", "ERR", "ERR", "ERR")
    except ValueError:
        pra_category, comp_type, description = ("ERR_VAL", "ERR_VAL", "ERR_VAL")

    file_handle.write(f"{pair_id},{arrival},{departure_time},{wait},{status},{match_type},{match_group},{epoch},{blood_p},{blood_d},{pra_category},{comp_type},{description}\n")
