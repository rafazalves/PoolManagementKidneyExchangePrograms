import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from kep.constants import *

POLICY_CONFIGS = {
    "hier_with_wt": {
        "folder": "kep_hier_with_wt",
        "name": "Hierarchical with WT"
    },
    "hier_no_wt": {
        "folder": "kep_hier_no_wt",
        "name": "Hierarchical No WT"
    },
    "max_cardinality": {
        "folder": "kep_max_cardinality",
        "name": "Max Cardinality"
    },
    "static": {
        "folder": "kep_static",
        "name": "Static"
    }
}

ACCEPTANCE_RATES = [0, 25, 50, 75, 100]
YEARS = [6, 12, 24]

BASE_RESULTS_DIR = "results"
BASE_OUTPUT_DIR = "processed_results"

def load_data(input_dir):
    """
    Load the data from results_*.csv and history_*.csv files.
    """
    print("Loading Data...")

    # Load Results Files (results_*.csv)
    result_files = glob.glob(os.path.join(input_dir, "results_*.csv"))
    print(f"Found {len(result_files)} results files.")

    results_list = []
    for f in result_files:
        try:
            df = pd.read_csv(f)
            results_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if results_list:
        results_df = pd.concat(results_list, ignore_index=True)
    else:
        results_df = pd.DataFrame()

    # Load History Files (history_*.csv)
    history_files = glob.glob(os.path.join(input_dir, "history_*.csv"))
    print(f"Found {len(history_files)} history files.")

    history_list = []
    for f in history_files:
        try:
            df = pd.read_csv(f)
            instance_id = int(os.path.basename(f).split("_")[1].split(".")[0])
            df["Instance"] = instance_id
            history_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if history_list:
        history_df = pd.concat(history_list, ignore_index=True)
    else:
        history_df = pd.DataFrame()

    return results_df, history_df

def load_matched_data(input_dir):
    """
    Load the data from matched_*.csv files.
    Expects lines in the format: "Epoch:ID1,ID2,ID3..."
    """
    print("Loading Matched Data...")
    files = glob.glob(os.path.join(input_dir, "matched_*.csv"))
    
    parsed_data = []
    
    for f in files:
        # Extract instance ID from filename (e.g., matched_0.csv -> 0)
        try:
            basename = os.path.basename(f)
            instance_id = int(basename.split("_")[1].split(".")[0])
        except:
            instance_id = 0 # Fallback
            
        with open(f, "r") as file:
            for line in file:
                line = line.strip()
                if not line or ":" not in line: continue
                
                parts = line.split(":")
                epoch = int(parts[0])
                ids_str = parts[1]

                parsed_data.append({
                    "Instance": instance_id,
                    "Epoch": epoch,
                    "MatchString": ids_str
                })
                
    return pd.DataFrame(parsed_data)

def analyze_performance(results_df, output_dir):
    """
    Analyse the solver"s performance (execution time).
    Generate two CSV files:
    1. solver_time_per_instance.csv: Average time per epoch for each instance.
    2. solver_time_summary.csv: Global average, standard deviation, and confidence interval.
    """
    
    # 1. Calcule the average solver time per epoch for each instance
    per_instance_df = results_df.groupby("Instance")["IterSolverTime"].mean().reset_index()
    per_instance_df.rename(columns={"IterSolverTime": "Avg_Solver_Time_Sec"}, inplace=True)
    
    per_instance_path = os.path.join(output_dir, "solver_time_per_instance.csv")
    per_instance_df.to_csv(per_instance_path, index=False)
    print(f"   -> Guardado: {per_instance_path}")

    # 2. Calculate global statistics (mean, std, confidence interval)
    mean_val = per_instance_df["Avg_Solver_Time_Sec"].mean()
    std_val = per_instance_df["Avg_Solver_Time_Sec"].std()
    count = len(per_instance_df)
    
    se = std_val / np.sqrt(count) # Standard Error
    ci_margin = 1.96 * se
    
    ci_lower = mean_val - ci_margin
    ci_upper = mean_val + ci_margin

    summary_data = {
        "Metric": ["Solver Time (s)"],
        "Mean": [mean_val],
        "Std_Dev": [std_val],
        "CI_Lower_95": [ci_lower],
        "CI_Upper_95": [ci_upper],
        "N_Instances": [count]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    summary_path = os.path.join(output_dir, "solver_time_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"   -> Guardado: {summary_path}")

def analyze_outcomes(history_df, output_dir):
    """
    Analyze the outcomes of the matches.
    Generate two CSV files:
    1. outcomes_per_instance.csv: Count of each status (MATCHED, IMMUNO, etc.) and their rates per instance.
    2. outcomes_summary.csv: Global statistics.
    """

    if history_df.empty:
        print("[WARN] No history data to analyze.")
        return

    print("Generating Pairs & Outcomes Analysis...")

    if "Description" in history_df.columns:
        df_pairs = history_df[history_df["Description"] != "altruist"].copy()
        print(f"(Altruists excluded. Analyzed Pairs: {len(df_pairs)})")
    else:
        df_pairs = history_df.copy()

    if df_pairs.empty:
        print("[WARN] History data is empty after filtering altruists.")
        return

    # 1. Calculate Outcomes per Instance
    outcomes_df = df_pairs.pivot_table(
        index="Instance", 
        columns="Status", 
        aggfunc="size", 
        fill_value=0
    )

    # Ensure that all expected status columns are present, even if they have zero counts
    expected_statuses = ["MATCHED", "IMMUNO", "COMPATIBLE_LEFT", "TIMEOUT", "UNMATCHED_END"]
    for status in expected_statuses:
        if status not in outcomes_df.columns:
            outcomes_df[status] = 0

    outcomes_df = outcomes_df[expected_statuses]

    # 2. Calculate Total Pairs and Rates
    outcomes_df["Total_Pairs"] = outcomes_df.sum(axis=1)

    # Rates for each status
    for status in expected_statuses:
        outcomes_df[f"{status}_Rate"] = outcomes_df[status] / outcomes_df["Total_Pairs"]

    # Save the per-instance outcomes data
    per_instance_path = os.path.join(output_dir, "outcomes_per_instance.csv")
    outcomes_df.to_csv(per_instance_path)
    print(f"-> Saved: {per_instance_path}")

    # 3. Calculate Global Statistics
    summary_list = []
    
    stats_cols = expected_statuses + ["Total_Pairs"] + [f"{s}_Rate" for s in expected_statuses]
    
    N = len(outcomes_df)

    for col in stats_cols:
        series = outcomes_df[col]
        mean_val = series.mean()
        std_val = series.std()
        
        se = std_val / np.sqrt(N)
        ci_margin = 1.96 * se
        
        summary_list.append({
            "Metric": col,
            "Mean": mean_val,
            "Std_Dev": std_val,
            "CI_Lower_95": mean_val - ci_margin,
            "CI_Upper_95": mean_val + ci_margin,
            "N_Instances": N
        })

    summary_df = pd.DataFrame(summary_list)
    
    summary_path = os.path.join(output_dir, "outcomes_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"-> Saved: {summary_path}")

def analyze_match_types(history_df, output_dir):
    """
    Analyze the structure of the matches:
    1. Cycle vs Chain (How many pairs were matched via cycles vs chains).
    2. Size of Matches (How many pairs in 2-way cycles vs 3-way cycles).
    """
    
    if history_df.empty: 
        return

    print("   -> Generating Match Structure Analysis (Cycles vs Chains)...")

    # Only keep "MATCHED" pairs for this analysis, and exclude altruists
    if "Description" in history_df.columns:
        df_matched = history_df[
            (history_df["Status"].isin(["MATCHED"])) & 
            (history_df["Description"] != "altruist")
        ].copy()
    else:
        df_matched = history_df[history_df["Status"].isin(["MATCHED"])].copy()

    if df_matched.empty:
        print("      [WARN] No matches found to analyze types.")
        return

    # 1. Pairs matched via cycles vs chains
    type_counts = df_matched.groupby(["Instance", "MatchType"]).size().unstack(fill_value=0)
    
    for col in ["Cycle", "Chain"]:
        if col not in type_counts.columns:
            type_counts[col] = 0
            
    type_counts["Total_Matched"] = type_counts["Cycle"] + type_counts["Chain"]
    type_counts["Cycle_Fraction"] = type_counts["Cycle"] / type_counts["Total_Matched"]
    type_counts["Chain_Fraction"] = type_counts["Chain"] / type_counts["Total_Matched"]

    type_path = os.path.join(output_dir, "match_types_per_instance.csv")
    type_counts.to_csv(type_path)

    # 2. Size of Matches (2-way vs 3-way)    
    size_data = []

    for instance_id, group in df_matched.groupby("Instance"):
        # Aggregate by Epoch + MatchType + MatchGroup to count how many pairs are in each cycle/chain
        sizes = group.groupby(["Epoch", "MatchType", "MatchGroup"]).size().reset_index(name="Size")
        
        cycles = sizes[sizes["MatchType"] == "Cycle"]
        chains = sizes[sizes["MatchType"] == "Chain"]
        
        # Count how many pairs are in 2-way cycles and how many in 3-way cycles
        pairs_in_2_cycles = cycles[cycles["Size"] == 2]["Size"].sum()
        pairs_in_3_cycles = cycles[cycles["Size"] == 3]["Size"].sum()
        
        # For chains, we can have variable sizes, so we just sum all pairs in chains
        pairs_in_chains = chains["Size"].sum()

        size_data.append({
            "Instance": instance_id,
            "Pairs_in_2_Cycles": pairs_in_2_cycles,
            "Pairs_in_3_Cycles": pairs_in_3_cycles,
            "Pairs_in_Chains": pairs_in_chains,
            "Total_Matched": pairs_in_2_cycles + pairs_in_3_cycles + pairs_in_chains
        })

    size_df = pd.DataFrame(size_data)

    size_path = os.path.join(output_dir, "match_sizes_per_instance.csv")
    size_df.to_csv(size_path, index=False)
    print(f"-> Saved: {size_path}")

    # 3. Global summary of match types and sizes
    summary_list = []
    
    # A. Resume Types (Cycle vs Chain)
    for col in ["Cycle", "Chain", "Chain_Fraction", "Cycle_Fraction"]:
        vals = type_counts[col]
        summary_list.append({
            "Metric": f"Type_{col}",
            "Mean": vals.mean(),
            "Std": vals.std(),
            "CI_Upper": vals.mean() + 1.96 * (vals.std() / np.sqrt(len(vals))),
            "CI_Lower": vals.mean() - 1.96 * (vals.std() / np.sqrt(len(vals)))
        })

    # B. Resume Sizes (2-way vs 3-way vs Chains)
    for col in ["Pairs_in_2_Cycles", "Pairs_in_3_Cycles", "Pairs_in_Chains"]:
        vals = size_df[col]
        summary_list.append({
            "Metric": f"Size_{col}",
            "Mean": vals.mean(),
            "Std": vals.std(),
            "CI_Upper": vals.mean() + 1.96 * (vals.std() / np.sqrt(len(vals))),
            "CI_Lower": vals.mean() - 1.96 * (vals.std() / np.sqrt(len(vals)))
        })

    summary_df = pd.DataFrame(summary_list)
    sum_path = os.path.join(output_dir, "match_structure_summary.csv")
    summary_df.to_csv(sum_path, index=False)
    print(f"-> Saved: {sum_path}")

def analyze_demographics(history_df, output_dir):
    """
    Analyze the demographics of the pairs (Compatibility, Blood, PRA)
    Generate 2 CSV files for each metric:
    1. demographics_<metric>_per_instance.csv
    2. demographics_<metric>_summary.csv
    """

    if history_df.empty:
        print("[WARN] No history data.")
        return

    print("Generating Demographics Analysis...")

    if "Description" in history_df.columns:
        df = history_df[history_df["Description"] != "altruist"].copy()
    else:
        df = history_df.copy()

    if df.empty:
        return

    # Map the compatibility codes to labels
    compat_map = {1: "Incompatible", 2: "Half-Compatible", 3: "Compatible"}
    if "Compatibility" in df.columns:
        df["Compat_Label"] = df["Compatibility"].map(compat_map)
    
    # Create binary columns for each status to facilitate counting
    df["is_matched"] = (df["Status"] == "MATCHED").astype(int)
    df["is_immuno"] = (df["Status"] == "IMMUNO").astype(int)
    df["is_timeout"] = (df["Status"] == "TIMEOUT").astype(int)

    # Auxiliary function to process each demographic group and generate the CSV files
    def _process_group(group_cols, filename_suffix):
        # A. Analyze per Instance
        gb = df.groupby(["Instance"] + group_cols)
        
        agg_df = gb.agg(
            Total=("PairID", "count"),
            Matched=("is_matched", "sum"),
            Immuno=("is_immuno", "sum"),
            Timeout=("is_timeout", "sum")
        ).reset_index()

        agg_df["KEP_MatchRate"] = agg_df["Matched"] / agg_df["Total"]
        agg_df["Total_TransplantRate"] = (agg_df["Matched"] + agg_df["Immuno"]) / agg_df["Total"]

        inst_path = os.path.join(output_dir, f"demographics_{filename_suffix}_per_instance.csv")
        agg_df.to_csv(inst_path, index=False)

        # B. Global Summary
        # Metrics to summarize
        metrics = ["Total", "Matched", "Immuno", "Timeout", "KEP_MatchRate", "Total_TransplantRate"]
        
        summary_list = []
        
        unique_groups = agg_df[group_cols].drop_duplicates()
        
        for _, row_group in unique_groups.iterrows():
            mask = pd.Series([True] * len(agg_df))
            for col in group_cols:
                mask = mask & (agg_df[col] == row_group[col])
            
            subset = agg_df[mask]
            n_instances = len(subset)

            record = {col: row_group[col] for col in group_cols}
            record["N_Instances"] = n_instances # 100 instances

            for m in metrics:
                vals = subset[m]
                mean_val = vals.mean()
                std_val = vals.std()
                
                if n_instances > 1:
                    se = std_val / np.sqrt(n_instances)
                    ci = 1.96 * se
                else:
                    ci = 0

                record[f"{m}_Mean"] = mean_val
                record[f"{m}_Std"] = std_val
                record[f"{m}_CI_Lower"] = mean_val - ci
                record[f"{m}_CI_Upper"] = mean_val + ci
            
            summary_list.append(record)
        
        summary_df = pd.DataFrame(summary_list)
        summary_df = summary_df.sort_values(by=group_cols)

        # Save Summary CSV
        sum_path = os.path.join(output_dir, f"demographics_{filename_suffix}_summary.csv")
        summary_df.to_csv(sum_path, index=False)
        print(f"-> Saved Analysis: {filename_suffix}")

    # ---------------------------------------------------------
    # Executions for different demographic groupings
    # ---------------------------------------------------------

    # 1. For Compatibility (Incompatible, Half-Compatible, Compatible)
    if "Compat_Label" in df.columns:
        _process_group(["Compat_Label"], "compatibility")

    # 2. For Blood Type (A, B, AB, O)
    if "BloodPatient" in df.columns:
        _process_group(["BloodPatient"], "blood")

    # 3. For PRA (Low, Medium, High)
    if "PRA" in df.columns:
        _process_group(["PRA"], "pra")

    # 4. For Blood + Compatibility (ex: Blood A & Incompatible)
    if "BloodPatient" in df.columns and "Compat_Label" in df.columns:
        _process_group(["BloodPatient", "Compat_Label"], "blood_and_compat")

    # 5. For PRA + Compatibility (ex: PRA High & Incompatible)
    if "PRA" in df.columns and "Compat_Label" in df.columns:
        _process_group(["PRA", "Compat_Label"], "pra_and_compat")

def analyze_pool_evolution(results_df, history_df, output_dir):
    """
    Analyze the temporal evolution of the Pool (by Epoch).
    Metrics to analyze:
    1. Total Pairs in Pool.
    2. Total Patient Blood Type O.
    3. Total Patient Blood Type O + Donor AB.
    4. Timeouts in each epoch.
    """

    if history_df.empty or results_df.empty:
        print("[WARN] Missing data for pool evolution analysis.")
        return

    print("Generating Pool Evolution Analysis...")

    if "Description" in history_df.columns:
        hist_pairs = history_df[history_df["Description"] != "altruist"].copy()
    else:
        hist_pairs = history_df.copy()

    if "Period" in results_df.columns:
        period_days = results_df["Period"].iloc[0]
    else:
        period_days = 90 

    evolution_data = []

    hist_grouped = hist_pairs.groupby("Instance")
    res_grouped = results_df.groupby("Instance")

    for instance_id, res_group in res_grouped:
        
        if instance_id not in hist_grouped.groups:
            continue
            
        hist_group = hist_grouped.get_group(instance_id)

        # Order by Epoch to ensure we process in the correct temporal order
        res_group = res_group.sort_values("Epoch")
        last_epoch = res_group["Epoch"].max()

        for _, row in res_group.iterrows():
            epoch = row["Epoch"]
            # Total pairs in the pool at the start of this epoch (NNodes)
            pool_total = row["NNodes"] 
            
            # Current day in the simulation corresponding to this epoch
            current_day = epoch * period_days

            previous_day = current_day - period_days

            # A. Calculate the composition of the pool at the start of this epoch
            if epoch == last_epoch:
                active_mask = (hist_group["Arrival"] <= current_day) & (hist_group["Departure"] >= current_day) & (hist_group["Status"] == "UNMATCHED_END")
            else:
                active_mask = (hist_group["Arrival"] <= current_day) & (hist_group["Departure"] > current_day)
            
            active_pairs = hist_group[active_mask]

            # Incompatible pairs (Compatibility == 1)
            count_incomp = active_pairs[active_pairs["Compatibility"] == 1].shape[0]

            # Patients with Blood Type O
            count_O = active_pairs[active_pairs["BloodPatient"] == "O"].shape[0]

            # Patients with Blood Type O + Donor with Blood Type AB
            count_O_AB = active_pairs[
                (active_pairs["BloodPatient"] == "O") & 
                (active_pairs["BloodDonor"] == "AB")
            ].shape[0]

            # B. Calculate how many timeouts occurred during this epoch (Departure between previous_day and current_day)
            timeouts_epoch = hist_group[
                (hist_group["Status"] == "TIMEOUT") & 
                (hist_group["Departure"] > previous_day) & 
                (hist_group["Departure"] <= current_day)
            ].shape[0]

            evolution_data.append({
                "Instance": instance_id,
                "Epoch": epoch,
                "Simulation_Day": current_day,
                "Pool_Total": pool_total,
                "Pool_Patient_O": count_O,
                "Pool_Patient_O_Donor_AB": count_O_AB,
                "Pool_Incompatible": count_incomp,
                "Timeouts_In_Epoch": timeouts_epoch
            })

    evo_df = pd.DataFrame(evolution_data)

    if evo_df.empty:
        print("[WARN] No evolution data generated.")
        return

    per_instance_path = os.path.join(output_dir, "pool_evolution_per_instance.csv")
    evo_df.to_csv(per_instance_path, index=False)

    # 2. Calculate global statistics by Epoch (mean, std, confidence interval)
    summary_list = []
        
    metrics = ["Pool_Total", "Pool_Patient_O", "Pool_Patient_O_Donor_AB", "Pool_Incompatible", "Timeouts_In_Epoch"]
    
    # Aggregate by Epoch
    grouped_epoch = evo_df.groupby("Epoch")

    for epoch, group in grouped_epoch:
        n_instances = len(group)
        record = {"Epoch": epoch, "Simulation_Day": group["Simulation_Day"].iloc[0], "N_Instances": n_instances}

        for m in metrics:
            vals = group[m]
            mean_val = vals.mean()
            std_val = vals.std()
            
            # IC 95%
            if n_instances > 1:
                se = std_val / np.sqrt(n_instances)
                ci = 1.96 * se
            else:
                ci = 0
            
            record[f"{m}_Mean"] = mean_val
            record[f"{m}_Std"] = std_val
            record[f"{m}_CI_Lower"] = mean_val - ci
            record[f"{m}_CI_Upper"] = mean_val + ci
        
        summary_list.append(record)

    summary_df = pd.DataFrame(summary_list)
    
    summary_path = os.path.join(output_dir, "pool_evolution_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"-> Saved: {summary_path}")

def analyze_waiting_times(history_df, output_dir):
    """
    Analyze the waiting times of the matched pairs.
    """
    if history_df.empty:
        print("[WARN] No history data for waiting times.")
        return

    print("Generating Waiting Times Analysis...")

    if "Description" in history_df.columns:
        df = history_df[history_df["Description"] != "altruist"].copy()
    else:
        df = history_df.copy()

    compat_map = {1: "Incompatible", 2: "Half-Compatible", 3: "Compatible"}
    if "Compatibility" in df.columns:
        df["Compat_Label"] = df["Compatibility"].map(compat_map)

    matched_mask = df["Status"].isin(["MATCHED"])
    df_matched = df[matched_mask].copy()

    if df_matched.empty:
        print("[WARN] No matches found. Skipping wait time analysis.")
        return

    # Auxiliary function to process waiting times for different groups
    def _process_wait_group(group_cols, filename_suffix):
        if not group_cols:
            gb = df_matched.groupby(["Instance"])
            group_keys = [] 
        else:
            gb = df_matched.groupby(["Instance"] + group_cols)
            group_keys = group_cols

        # A. Per Instance Analysis
        agg_df = gb.agg(
            Count=("WaitTime", "count"),
            Mean_Wait=("WaitTime", "mean"),
            Median_Wait=("WaitTime", "median"),
            Max_Wait=("WaitTime", "max")
        ).reset_index()

        inst_path = os.path.join(output_dir, f"waittime_{filename_suffix}_per_instance.csv")
        agg_df.to_csv(inst_path, index=False)

        # B. Global Summary
        summary_list = []
        
        if not group_keys:
            unique_groups = [None]
        else:
            unique_groups = agg_df[group_keys].drop_duplicates()
            unique_groups = [row.to_dict() for _, row in unique_groups.iterrows()]

        for group_dict in unique_groups:
            if group_dict is None:
                subset = agg_df
                record = {"Group": "All Matched"}
            else:
                record = {}
                mask = pd.Series([True] * len(agg_df))
                for k, v in group_dict.items():
                    mask = mask & (agg_df[k] == v)
                    record[k] = v
                subset = agg_df[mask]

            n_instances = len(subset)
            record["N_Instances_With_Matches"] = n_instances

            metrics_to_summarize = ["Mean_Wait", "Median_Wait", "Max_Wait"]
            
            for m in metrics_to_summarize:
                vals = subset[m]
                mean_val = vals.mean()
                std_val = vals.std()
                
                # IC 95%
                if n_instances > 1:
                    se = std_val / np.sqrt(n_instances)
                    ci = 1.96 * se
                else:
                    ci = 0

                record[f"{m}_GlobalAvg"] = mean_val
                record[f"{m}_Std"] = std_val
                record[f"{m}_CI_Lower"] = mean_val - ci
                record[f"{m}_CI_Upper"] = mean_val + ci

            summary_list.append(record)

        summary_df = pd.DataFrame(summary_list)
        
        if group_keys:
            summary_df = summary_df.sort_values(by=group_keys)

        sum_path = os.path.join(output_dir, f"waittime_{filename_suffix}_summary.csv")
        summary_df.to_csv(sum_path, index=False)
        print(f"      -> Saved Waiting Time Analysis: {filename_suffix}")


    # 1. General (All Matched Pairs)
    _process_wait_group([], "general")

    # 2. By Compatibility
    if "Compat_Label" in df.columns:
        _process_wait_group(["Compat_Label"], "compatibility")

    # 3. By Blood Type divided by Compatibility
    if "BloodPatient" in df.columns and "Compat_Label" in df.columns:
        _process_wait_group(["BloodPatient", "Compat_Label"], "blood_and_compat")

    # 4. PRA divided by Compatibility
    if "PRA" in df.columns and "Compat_Label" in df.columns:
        _process_wait_group(["PRA", "Compat_Label"], "pra_and_compat")

def analyze_waiting_times_top_10_percent(history_df, output_dir):
    """
    Analyze the waiting times for the top 10% of pairs with the longest waiting times (Worst Case).
    
    Generate 2 CSV files:
    1. waittime_top10_global_per_instance.csv: Detailed data for each instance.
    2. waittime_top10_global_summary.csv: Global summary.
    """
    
    if history_df.empty:
        print("[WARN] No history data provided.")
        return

    print("Generating Waiting Times Analysis (Top 10% Worst Cases)...")

    if "Description" in history_df.columns:
        df = history_df[history_df["Description"] != "altruist"].copy()
    else:
        df = history_df.copy()

    # Only analyze the waiting times of those who were actually transplanted (MATCHED)
    df = df[df["Status"] == "MATCHED"]

    if df.empty:
        print("[WARN] No matched pairs found for Top 10% analysis.")
        return

    per_instance_stats = []

    # 1. Processing by Instance
    for instance_id, group in df.groupby("Instance"):
        if group.empty:
            continue

        # Organize the group by WaitTime in descending order to get the worst cases at the top
        group_sorted = group.sort_values(by="WaitTime", ascending=False)
        
        # Count how many pairs are in this instance
        total_pairs = len(group_sorted)
        
        # Calculate how many pairs correspond to the top 10%
        n_top = int(np.ceil(total_pairs * 0.10))
        
        if n_top == 0:
            continue

        # Get the subset of the top 10% worst cases for this instance
        top_10_subset = group_sorted.head(n_top)
        
        s_sum = top_10_subset["WaitTime"].sum()
        s_mean = top_10_subset["WaitTime"].mean()
        s_count = len(top_10_subset)
        s_max = top_10_subset["WaitTime"].max()
        s_min = top_10_subset["WaitTime"].min()
        
        # Store the statistics for this instance
        per_instance_stats.append({
            "Instance": instance_id,
            "Instance_Waiting_Time_Mean": s_mean,
            "Top10_Sum": s_sum,
            "Top10_Count": s_count,
            "Max_WaitTime": s_max,
            "Min_WaitTime": s_min
        })

    if not per_instance_stats:
        print("   [WARN] Could not calculate averages (no data in top 10%).")
        return

    df_per_instance = pd.DataFrame(per_instance_stats)
    df_per_instance = df_per_instance.sort_values(by="Instance")

    detailed_path = os.path.join(output_dir, "waittime_top10_global_per_instance.csv")
    df_per_instance.to_csv(detailed_path, index=False)
    print(f"-> Saved (Instance Details): {detailed_path}")

    # 2. Calculate global summary statistics based on the per-instance means of the top 10% worst cases
    global_mean = df_per_instance["Instance_Waiting_Time_Mean"].mean()
    sum_of_means = df_per_instance["Instance_Waiting_Time_Mean"].sum()
    n_instances = len(df_per_instance)
    global_max = df_per_instance["Max_WaitTime"].max()
    global_min = df_per_instance["Min_WaitTime"].min()

    std_val = df_per_instance["Instance_Waiting_Time_Mean"].std()
    se = std_val / np.sqrt(n_instances) if n_instances > 1 else 0
    ci_margin = 1.96 * se
    ci_lower = global_mean - ci_margin
    ci_upper = global_mean + ci_margin

    summary_data = {
        "Global_Mean": [global_mean],
        "Global_Mean_CI_Lower": [ci_lower],
        "Global_Mean_CI_Upper": [ci_upper],
        "Sum_of_Instance_Means": [sum_of_means],
        "N_Instances_Processed": [n_instances],
        "Max_Wait_Global": [global_max],
        "Min_Wait_Global": [global_min]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    summary_path = os.path.join(output_dir, "waittime_top10_global_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"-> Saved (Global Summary): {summary_path}")

def analyze_waiting_times_top10_by_compatibility(history_df, output_dir):
    """
    Analyze the waiting times for the top 10% of pairs with the longest waiting times (Worst Case),
    by Compatibility Type.

    Generates 2 CSV files:
    - waittime_top10_compatibility_per_instance.csv
    - waittime_top10_compatibility_summary.csv
    """
    
    if history_df.empty:
        print("[WARN] No history data provided.")
        return

    print("Generating Waiting Times Analysis by Compatibility (Top 10% Worst Cases)...")

    if "Description" in history_df.columns:
        df = history_df[history_df["Description"] != "altruist"].copy()
    else:
        df = history_df.copy()

    # Only keep MATCHED pairs for this analysis
    df = df[df["Status"] == "MATCHED"]

    if df.empty:
        print("[WARN] No matched pairs found for analysis.")
        return

    # Map the compatibility codes to labels for better readability
    comp_map = {1: "Incompatible", 2: "Half-Compatible", 3: "Compatible"}
    
    per_instance_stats = []

    # 1. Processing by Instance and Compatibility Type
    df = df[df["Compatibility"].isin([1, 2, 3])]

    for (instance_id, comp_type), group in df.groupby(["Instance", "Compatibility"]):
        if group.empty:
            continue

        # Order the group by WaitTime in descending order to get the worst cases at the top
        group_sorted = group.sort_values(by="WaitTime", ascending=False)
        
        # Count how many pairs exist
        total_pairs = len(group_sorted)
        
        # Calculate 10% of the pairs
        n_top = int(np.ceil(total_pairs * 0.10))
        
        if n_top == 0:
            continue

        # Get the subset of the top 10% worst cases for this instance and compatibility type
        top_10_subset = group_sorted.head(n_top)
        
        s_sum = top_10_subset["WaitTime"].sum()
        s_mean = top_10_subset["WaitTime"].mean()
        s_count = len(top_10_subset)
        s_max = top_10_subset["WaitTime"].max()
        s_min = top_10_subset["WaitTime"].min()
        
        comp_label = comp_map.get(comp_type, f"Type_{comp_type}")

        per_instance_stats.append({
            "Instance": instance_id,
            "Compatibility_ID": comp_type,
            "Compatibility": comp_label,
            "Instance_Mean": s_mean,
            "Top10_Sum": s_sum,
            "Top10_Count": s_count,
            "Max_WaitTime": s_max,
            "Min_WaitTime": s_min
        })

    if not per_instance_stats:
        print("[WARN] Could not calculate averages (no data found).")
        return

    df_per_instance = pd.DataFrame(per_instance_stats)
    df_per_instance = df_per_instance.sort_values(by=["Instance", "Compatibility_ID"])

    detailed_path = os.path.join(output_dir, "waittime_top10_compatibility_per_instance.csv")
    df_per_instance.to_csv(detailed_path, index=False)
    print(f"-> Saved (Detailed): {detailed_path}")

    # 2. Calculation of Global Summary
    summary_list = []

    # Iterate through each compatibility type to calculate global statistics
    for comp_id, group_res in df_per_instance.groupby("Compatibility_ID"):
        
        comp_label = group_res["Compatibility"].iloc[0]
        
        global_mean = group_res["Instance_Mean"].mean()
        sum_of_means = group_res["Instance_Mean"].sum()
        n_instances = len(group_res)
        
        global_max = group_res["Max_WaitTime"].max()
        global_min = group_res["Min_WaitTime"].min()

        std_val = group_res["Instance_Mean"].std()
        se = std_val / np.sqrt(n_instances) if n_instances > 1 else 0
        ci_margin = 1.96 * se
        ci_lower = global_mean - ci_margin
        ci_upper = global_mean + ci_margin

        summary_list.append({
            "Compatibility_ID": comp_id,
            "Compatibility": comp_label,
            "Global_Mean": global_mean,
            "Global_Mean_CI_Lower": ci_lower,
            "Global_Mean_CI_Upper": ci_upper,
            "Sum_of_Instance_Means": sum_of_means,
            "N_Instances_Processed": n_instances,
            "Max_Wait_Global": global_max,
            "Min_Wait_Global": global_min
        })

    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values(by="Compatibility_ID")

    summary_path = os.path.join(output_dir, "waittime_top10_compatibility_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"-> Saved (Global Summary): {summary_path}")

def analyze_blood_O_leakage(history_df, matched_df, output_dir):
    """
    Analyze the flow of Blood Type O Donors. Generates 3 levels of detail:
    1. Raw: List of all donations.
    2. Per Instance: Summary by instance (Count & %).
    3. Summary: Global average + Confidence Interval (95%).
    """
    print("Analyzing Blood O Donor flow (with Donor Compatibility)...")

    df_analysis = history_df.copy()
    if "PairID" in df_analysis.columns and "ID" not in df_analysis.columns:
        df_analysis = df_analysis.rename(columns={"PairID": "ID"})

    justif_dir = os.path.join(output_dir, "justification_results")
    if not os.path.exists(justif_dir):
        os.makedirs(justif_dir)

    lookup = {}
    compat_map = {1: "Incompatible", 2: "Half-Compatible", 3: "Compatible"}
    
    if "Compatibility" not in df_analysis.columns:
        print("[WARN] 'Compatibility' missing. Cannot analyze.")
        return

    # List of all instances for later use in creating the full grid
    all_instances = sorted(df_analysis["Instance"].unique())

    for _, row in df_analysis.iterrows():
        inst = row["Instance"]
        pid = row["ID"]
        if inst not in lookup: lookup[inst] = {}
        
        c_val = row["Compatibility"]
        c_label = compat_map.get(c_val, f"Unknown({c_val})")

        lookup[inst][pid] = {
            "BloodDonor": row["BloodDonor"],
            "BloodPatient": row["BloodPatient"],
            "CompatLabel": c_label,
            "IsAltruist": (row["Description"] == "altruist") if "Description" in row else False
        }

    # 1. Process the matched results to extract the flow of Blood O donors
    leakage_records = []
    
    for _, row in matched_df.iterrows():
        inst = row["Instance"]
        ids_str = row["MatchString"]
        
        if not ids_str: continue
        try:
            ids = [int(x) for x in ids_str.split(",")]
        except ValueError: continue
        if len(ids) < 2: continue 
        
        if inst not in lookup: continue
        first_id = ids[0]
        if first_id not in lookup[inst]: continue
        
        is_chain = lookup[inst][first_id]["IsAltruist"]
        
        donations = []
        for i in range(len(ids) - 1):
            donations.append((ids[i], ids[i+1]))
        if not is_chain:
            donations.append((ids[-1], ids[0]))

        for donor_id, recipient_id in donations:
            if donor_id not in lookup[inst] or recipient_id not in lookup[inst]: continue
            
            donor_info = lookup[inst][donor_id]
            recipient_info = lookup[inst][recipient_id]
            
            # Only analyze if the donor is Blood Type O
            if donor_info["BloodDonor"] == "O":
                recip_blood = recipient_info["BloodPatient"]
                recip_compat = recipient_info["CompatLabel"]
                
                donor_compat = donor_info["CompatLabel"]
                
                match_type = "Identical (O->O)" if recip_blood == "O" else f"Leakage (O->{recip_blood})"
                
                leakage_records.append({
                    "Instance": inst,
                    "DonorID": donor_id,
                    "RecipientID": recipient_id,
                    "DonorBlood": "O",
                    "DonorCompat": donor_compat,
                    "RecipientBlood": recip_blood,
                    "RecipientCompat": recip_compat,
                    "MatchType": match_type
                })

    # Output 1: Raw Data of Blood O donations (with compatibility info)
    df_leak = pd.DataFrame(leakage_records)
    if df_leak.empty:
        print("      [WARN] No Blood O donations found.") 
        return

    raw_path = os.path.join(justif_dir, "blood_O_flow_detailed_raw.csv")
    df_leak.to_csv(raw_path, index=False)

    # Output 2: Per Instance (Summary by Instance)
    group_cols = ["MatchType", "DonorCompat", "RecipientCompat"]
    
    counts = df_leak.groupby(["Instance"] + group_cols).size().reset_index(name="Count")

    unique_types = counts[group_cols].drop_duplicates()
    df_all_inst = pd.DataFrame({"Instance": all_instances})
    
    df_all_inst["key"] = 1
    unique_types["key"] = 1
    full_grid = pd.merge(df_all_inst, unique_types, on="key").drop("key", axis=1)
    
    per_inst = pd.merge(full_grid, counts, on=["Instance"] + group_cols, how="left").fillna(0)
    
    # Rates: For each instance, calculate the percentage of each MatchType relative to the total Blood O donations in that instance
    inst_totals = per_inst.groupby("Instance")["Count"].transform("sum")
    per_inst["Percentage"] = per_inst.apply(
        lambda x: (x["Count"] / inst_totals[x.name] * 100) if inst_totals[x.name] > 0 else 0, axis=1
    )

    per_inst_path = os.path.join(justif_dir, "blood_O_flow_per_instance.csv")
    per_inst.to_csv(per_inst_path, index=False)

    # Output 3: Summary with Global Averages and Confidence Intervals
    summary = per_inst.groupby(group_cols).agg(
        Mean_Count=("Count", "mean"),
        Std_Count=("Count", "std"),
        Mean_Percentage=("Percentage", "mean"),
        Std_Percentage=("Percentage", "std"),
        N_Instances=("Instance", "count")
    ).reset_index()
    
    summary["Count_SE"] = summary["Std_Count"] / np.sqrt(summary["N_Instances"])
    summary["Count_CI_Lower"] = summary["Mean_Count"] - (1.96 * summary["Count_SE"])
    summary["Count_CI_Upper"] = summary["Mean_Count"] + (1.96 * summary["Count_SE"])
    
    summary["Percentage_SE"] = summary["Std_Percentage"] / np.sqrt(summary["N_Instances"])
    summary["Percentage_CI_Lower"] = summary["Mean_Percentage"] - (1.96 * summary["Percentage_SE"])
    summary["Percentage_CI_Upper"] = summary["Mean_Percentage"] + (1.96 * summary["Percentage_SE"])

    summary["Count_CI_Lower"] = summary["Count_CI_Lower"].clip(lower=0)
    summary["Percentage_CI_Lower"] = summary["Percentage_CI_Lower"].clip(lower=0)
    summary = summary.sort_values(by="Mean_Count", ascending=False)

    summary_path = os.path.join(justif_dir, "blood_O_flow_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"-> Analysis Saved:")
    print(f"1. {raw_path}")
    print(f"2. {per_inst_path}")
    print(f"3. {summary_path}")

def analyze_incompatible_kidney_sources(history_df, matched_df, output_dir):
    """
    Analyze the sources of kidneys received by Incompatible pairs (Compatibility = 1) that were MATCHED.
    """
    if history_df.empty or matched_df.empty:
        return
    
    print("-> Analyzing sources of kidneys for Incompatible pairs...")

    df_analysis = history_df.copy()
    if "PairID" in df_analysis.columns and "ID" not in df_analysis.columns:
        df_analysis = df_analysis.rename(columns={"PairID": "ID"})

    justif_dir = os.path.join(output_dir, "justification_results")
    if not os.path.exists(justif_dir):
        os.makedirs(justif_dir)

    # Construct Lookup Dict
    lookup = {}
    compat_map = {
        1: "Incompatible", 
        2: "Half-Compatible", 
        3: "Compatible", 
        4: "Altruist"
    }
    
    all_instances = sorted(df_analysis["Instance"].unique())

    for _, row in df_analysis.iterrows():
        inst = row["Instance"]
        pid = row["ID"]
        if inst not in lookup: 
            lookup[inst] = {}
        
        # Read the compatibility value, defaulting to -1 if missing
        c_val = row.get("Compatibility", -1)
        
        c_label = compat_map.get(c_val, f"Unknown({c_val})")

        # Altruist has code 4
        is_altruist = (c_val == 4)

        lookup[inst][pid] = {
            "CompatLabel": c_label,
            "IsAltruist": is_altruist
        }

    # Get the sources of kidneys received by Incompatible pairs in the matched results
    source_records = []
    
    for _, row in matched_df.iterrows():
        inst = row["Instance"]
        ids_str = row["MatchString"]
        
        if not ids_str or pd.isna(ids_str): continue
        try:
            ids = [int(x) for x in str(ids_str).split(",")]
        except ValueError: continue
        if len(ids) < 2: continue 
        if inst not in lookup: continue
        
        first_id = ids[0]
        if first_id not in lookup[inst]: continue
        
        is_chain = lookup[inst][first_id]["IsAltruist"]
        
        donations = []
        for i in range(len(ids) - 1):
            donations.append((ids[i], ids[i+1]))
        if not is_chain:
            donations.append((ids[-1], ids[0]))

        for donor_id, recipient_id in donations:
            if donor_id not in lookup[inst] or recipient_id not in lookup[inst]: continue
            
            donor_info = lookup[inst][donor_id]
            recipient_info = lookup[inst][recipient_id]
            
            # The receiver must be Incompatible for us to analyze the source of their kidney
            if recipient_info["CompatLabel"] == "Incompatible":
                source_records.append({
                    "Instance": inst,
                    "DonorID": donor_id,
                    "RecipientID": recipient_id,
                    "DonorSource_Compat": donor_info["CompatLabel"]
                })

    # Output 1: Raw Data of Incompatible pairs and their kidney sources
    df_source = pd.DataFrame(source_records)
    if df_source.empty:
        print("      [WARN] No matches for Incompatible pairs found.")
        return

    df_source = df_source.sort_values(by=["Instance", "RecipientID"])

    raw_path = os.path.join(justif_dir, "incompatible_kidney_sources_raw.csv")
    df_source.to_csv(raw_path, index=False)

    # Output 2: Per Instance (Summary by Instance)
    group_cols = ["DonorSource_Compat"]
    counts = df_source.groupby(["Instance"] + group_cols).size().reset_index(name="Count")

    unique_types = pd.DataFrame({
        "DonorSource_Compat": ["Altruist", "Compatible", "Half-Compatible", "Incompatible"]
    })
    
    df_all_inst = pd.DataFrame({"Instance": all_instances})
    df_all_inst["key"] = 1
    unique_types["key"] = 1
    full_grid = pd.merge(df_all_inst, unique_types, on="key").drop("key", axis=1)
    
    per_inst = pd.merge(full_grid, counts, on=["Instance"] + group_cols, how="left").fillna(0)
    
    # Get rates for each instance
    inst_totals = per_inst.groupby("Instance")["Count"].transform("sum")
    per_inst["Percentage"] = per_inst.apply(
        lambda x: (x["Count"] / inst_totals[x.name] * 100) if inst_totals[x.name] > 0 else 0, axis=1
    )

    per_inst_path = os.path.join(justif_dir, "incompatible_kidney_sources_per_instance.csv")
    per_inst.to_csv(per_inst_path, index=False)

    # Output 3: Global Summary
    summary = per_inst.groupby(group_cols).agg(
        Mean_Count=("Count", "mean"),
        Std_Count=("Count", "std"),
        Mean_Percentage=("Percentage", "mean"),
        Std_Percentage=("Percentage", "std"),
        N_Instances=("Instance", "count")
    ).reset_index()

    z_score = 1.96
    
    summary["Count_SE"] = summary["Std_Count"] / np.sqrt(summary["N_Instances"])
    summary["Count_CI_Lower"] = (summary["Mean_Count"] - (z_score * summary["Count_SE"])).clip(lower=0)
    summary["Count_CI_Upper"] = summary["Mean_Count"] + (z_score * summary["Count_SE"])
    
    summary["Percentage_SE"] = summary["Std_Percentage"] / np.sqrt(summary["N_Instances"])
    summary["Percentage_CI_Lower"] = (summary["Mean_Percentage"] - (z_score * summary["Percentage_SE"])).clip(lower=0)
    summary["Percentage_CI_Upper"] = summary["Mean_Percentage"] + (z_score * summary["Percentage_SE"])

    summary = summary.sort_values(by="Mean_Count", ascending=False)

    summary_path = os.path.join(justif_dir, "incompatible_kidney_sources_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"-> Analysis Saved in justif_dir:")
    print(f"1. {os.path.basename(raw_path)}")
    print(f"2. {os.path.basename(per_inst_path)}")
    print(f"3. {os.path.basename(summary_path)}")

def analyze_high_pra_incomp_kidney_sources(history_df, matched_df, output_dir):
    """
    Analyze the sources of kidneys received by Incompatible pairs (Compatibility = 1) with High PRA that were MATCHED.
    """

    if history_df.empty or matched_df.empty:
        return
    
    if "PRA" not in history_df.columns:
        print("[WARN] 'PRA' column missing in history data. Cannot analyze High PRA sources.")
        return
    
    print("-> Analyzing sources of kidneys for Incompatible pairs with High PRA...")

    df_analysis = history_df.copy()
    if "PairID" in df_analysis.columns and "ID" not in df_analysis.columns:
        df_analysis = df_analysis.rename(columns={"PairID": "ID"})

    justif_dir = os.path.join(output_dir, "justification_results")
    if not os.path.exists(justif_dir):
        os.makedirs(justif_dir)

    lookup = {}
    compat_map = {
        1: "Incompatible", 
        2: "Half-Compatible", 
        3: "Compatible", 
        4: "Altruist"
    }
    
    all_instances = sorted(df_analysis["Instance"].unique())

    for _, row in df_analysis.iterrows():
        inst = row["Instance"]
        pid = row["ID"]
        if inst not in lookup: 
            lookup[inst] = {}
        
        # Read the compatibility value, defaulting to -1 if missing
        c_val = row.get("Compatibility", -1)
        
        c_label = compat_map.get(c_val, f"Unknown({c_val})")

        # Altruist has code 4
        is_altruist = (c_val == 4)

        pra_level = str(row.get("PRA", "")).strip().capitalize()

        lookup[inst][pid] = {
            "CompatLabel": c_label,
            "IsAltruist": is_altruist,
            "PRA": pra_level
        }

    # Get the sources of kidneys received by Incompatible pairs in the matched results
    source_records = []
    
    for _, row in matched_df.iterrows():
        inst = row["Instance"]
        ids_str = row["MatchString"]
        
        if not ids_str or pd.isna(ids_str): continue
        try:
            ids = [int(x) for x in str(ids_str).split(",")]
        except ValueError: continue
        if len(ids) < 2: continue 
        if inst not in lookup: continue
        
        first_id = ids[0]
        if first_id not in lookup[inst]: continue
        
        is_chain = lookup[inst][first_id]["IsAltruist"]
        
        donations = []
        for i in range(len(ids) - 1):
            donations.append((ids[i], ids[i+1]))
        if not is_chain:
            donations.append((ids[-1], ids[0]))

        for donor_id, recipient_id in donations:
            if donor_id not in lookup[inst] or recipient_id not in lookup[inst]: continue
            
            donor_info = lookup[inst][donor_id]
            recipient_info = lookup[inst][recipient_id]
            
            # The receiver must be Incompatible for us to analyze the source of their kidney
            if recipient_info["CompatLabel"] == "Incompatible" and recipient_info["PRA"] == "High":
                source_records.append({
                    "Instance": inst,
                    "DonorID": donor_id,
                    "RecipientID": recipient_id,
                    "DonorSource_Compat": donor_info["CompatLabel"]
                })

    # Output 1: Raw Data of Incompatible pairs with High PRA and their kidney sources
    df_source = pd.DataFrame(source_records)
    if df_source.empty:
        print("      [WARN] No matches for Incompatible pairs with High PRA found.")
        return

    df_source = df_source.sort_values(by=["Instance", "RecipientID"])

    raw_path = os.path.join(justif_dir, "high_pra_incomp_kidney_sources_raw.csv")
    df_source.to_csv(raw_path, index=False)

    # Output 2: Per Instance (Summary by Instance)
    group_cols = ["DonorSource_Compat"]
    counts = df_source.groupby(["Instance"] + group_cols).size().reset_index(name="Count")

    unique_types = pd.DataFrame({
        "DonorSource_Compat": ["Altruist", "Compatible", "Half-Compatible", "Incompatible"]
    })
    
    df_all_inst = pd.DataFrame({"Instance": all_instances})
    df_all_inst["key"] = 1
    unique_types["key"] = 1
    full_grid = pd.merge(df_all_inst, unique_types, on="key").drop("key", axis=1)
    
    per_inst = pd.merge(full_grid, counts, on=["Instance"] + group_cols, how="left").fillna(0)
    
    # Get rates for each instance
    inst_totals = per_inst.groupby("Instance")["Count"].transform("sum")
    per_inst["Percentage"] = per_inst.apply(
        lambda x: (x["Count"] / inst_totals[x.name] * 100) if inst_totals[x.name] > 0 else 0, axis=1
    )

    per_inst_path = os.path.join(justif_dir, "high_pra_incomp_kidney_sources_per_instance.csv")
    per_inst.to_csv(per_inst_path, index=False)

    # Output 3: Global Summary
    summary = per_inst.groupby(group_cols).agg(
        Mean_Count=("Count", "mean"),
        Std_Count=("Count", "std"),
        Mean_Percentage=("Percentage", "mean"),
        Std_Percentage=("Percentage", "std"),
        N_Instances=("Instance", "count")
    ).reset_index()

    z_score = 1.96
    
    summary["Count_SE"] = summary["Std_Count"] / np.sqrt(summary["N_Instances"])
    summary["Count_CI_Lower"] = (summary["Mean_Count"] - (z_score * summary["Count_SE"])).clip(lower=0)
    summary["Count_CI_Upper"] = summary["Mean_Count"] + (z_score * summary["Count_SE"])
    
    summary["Percentage_SE"] = summary["Std_Percentage"] / np.sqrt(summary["N_Instances"])
    summary["Percentage_CI_Lower"] = (summary["Mean_Percentage"] - (z_score * summary["Percentage_SE"])).clip(lower=0)
    summary["Percentage_CI_Upper"] = summary["Mean_Percentage"] + (z_score * summary["Percentage_SE"])

    summary = summary.sort_values(by="Mean_Count", ascending=False)

    summary_path = os.path.join(justif_dir, "high_pra_incomp_kidney_sources_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"-> Analysis Saved in justif_dir:")
    print(f"1. {os.path.basename(raw_path)}")
    print(f"2. {os.path.basename(per_inst_path)}")
    print(f"3. {os.path.basename(summary_path)}")

def analyze_outcomes_binned_histogram_incompatible(history_df, output_dir):
    """
    Generate data for the line graph (frequency histogram) with 90-day bins,
    ONLY for Incompatible pairs (Compatibility = 1).
    Categories:
    1. MATCHED
    2. UNMATCHED_END
    3. TIMEOUT
    """
    print("Generating Binned Histogram Data (Incompatible Only)...")

    if history_df.empty:
        return

    if "Description" in history_df.columns:
        df = history_df[history_df["Description"] != "altruist"].copy()
    else:
        df = history_df.copy()

    # Filter only Incompatible pairs (Compatibility == 1)
    if "Compatibility" not in df.columns:
        print("      [WARN] 'Compatibility' column missing. Cannot filter incompatible pairs.")
        return
        
    df = df[df["Compatibility"] == 1].copy()

    # Define bins from 0 to 24 years (in days) with a step of 90 days
    max_days = 24 * 360
    bins = np.arange(0, max_days + 91, 90)

    n_instances = history_df["Instance"].nunique()
    if n_instances == 0: n_instances = 1

    # Separate the DataFrame into the three categories based on "Status"
    df_matched = df[df["Status"] == "MATCHED"].copy()
    df_unmatched = df[df["Status"] == "UNMATCHED_END"].copy()
    df_timeout = df[df["Status"] == "TIMEOUT"].copy()

    # Auxiliary function to get binned counts for a given subset of data
    def get_binned_counts(sub_df, bins):
        if sub_df.empty:
            return pd.Series(0, index=bins[:-1])
        return pd.cut(sub_df["WaitTime"], bins=bins, labels=bins[:-1], include_lowest=True).value_counts().sort_index()

    counts_matched = get_binned_counts(df_matched, bins)
    counts_unmatched = get_binned_counts(df_unmatched, bins)
    counts_timeout = get_binned_counts(df_timeout, bins)

    hist_df = pd.DataFrame({
        "Bin_Start_Day": counts_matched.index,
        "Count_Matched": counts_matched.values / n_instances,
        "Count_Unmatched": counts_unmatched.values / n_instances,
        "Count_Timeout": counts_timeout.values / n_instances
    })

    file_path = os.path.join(output_dir, "histogram_outcomes_binned_incompatible.csv")
    hist_df.to_csv(file_path, index=False)
    print(f"-> Saved: {file_path}")

def run_analysis(solver_key):
    # Verify if the provided solver key exists in the configuration
    if solver_key not in POLICY_CONFIGS:
        print(f"Solver key '{solver_key}' not found in configuration.")
        return
    
    config = POLICY_CONFIGS[solver_key]
    solver_name = config["name"]
    solver_folder = config["folder"]

    print(f"==========================================")
    print(f"Starting Analysis for Solver: {solver_name}")
    print(f"==========================================")

    # 1. Loop for each year scenario
    for year in YEARS:
        # Skip "static" solver for Year 24 (didn't run it for 24 years)
        if solver_key == "static" and year == 24:
            print(f"Skipping execution for {solver_key} at {year} years.")
            continue

        print(f"\n>>> Analyzing Year Scenario: {year} Years <<<")

        # 2. Loop for each acceptance rate (0, 25, 50, 75, 100)
        for acceptance in ACCEPTANCE_RATES:
            
            folder_relative_path = f"{year}_year_simulation/{solver_folder}/{acceptance}_acceptance"
            input_dir = os.path.join(BASE_RESULTS_DIR, folder_relative_path)
            output_dir = os.path.join(BASE_OUTPUT_DIR, folder_relative_path)

            if not os.path.exists(input_dir):
                print(f"[SKIP] Folder not found: {input_dir}")
                continue

            print(f"\nProcessing: {year} Years | {acceptance}% Acceptance")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Load data
            results_df, history_df = load_data(input_dir)
            matched_df = load_matched_data(input_dir)

            if results_df.empty or history_df.empty or matched_df.empty:
                print(f"[WARN] No data found for {acceptance}%. Skipping.")
                continue
            
            print("Running Standard Analysis...")

            analyze_performance(results_df, output_dir)
            analyze_outcomes(history_df, output_dir)
            analyze_match_types(history_df, output_dir)
            analyze_demographics(history_df, output_dir)

            analyze_blood_O_leakage(history_df, matched_df, output_dir)

            analyze_incompatible_kidney_sources(history_df, matched_df, output_dir)
            analyze_high_pra_incomp_kidney_sources(history_df, matched_df, output_dir)

            if solver_key != "static":
                analyze_waiting_times_top_10_percent(history_df, output_dir)
                analyze_waiting_times_top10_by_compatibility(history_df, output_dir)

                analyze_waiting_times(history_df, output_dir)

                analyze_outcomes_binned_histogram_incompatible(history_df, output_dir)
                
                analyze_pool_evolution(results_df, history_df, output_dir)
            else:
                print("Skipping Pool Evolution for 'static' solver.")

            # ===========================================================================================
            # EXTRA ANALYSIS: (Only for Year 24) AFTER WARM-UP, FIRST 20 EPOCHS & LAST 20 EPOCHS
            # ===========================================================================================
            if year == 24 and solver_key != "static":
                # AFTER WARM-UP
                warm_up_epoch = 25
                print(f"\n [EXTRA] Running 'After Warm-Up' Analysis (Epochs > {warm_up_epoch})...")

                warmup_output_dir = os.path.join(output_dir, "after_warm_up")
                if not os.path.exists(warmup_output_dir):
                    os.makedirs(warmup_output_dir)

                # Filter Data
                results_wu = results_df[results_df["Epoch"] > warm_up_epoch].copy()
                
                if "Epoch" in history_df.columns:
                    history_wu = history_df[history_df["Epoch"] > warm_up_epoch].copy()
                else:
                    history_wu = history_df.copy() # Fallback

                if not results_wu.empty and not history_wu.empty:
                    analyze_performance(results_wu, warmup_output_dir)
                    analyze_outcomes(history_wu, warmup_output_dir)
                    analyze_match_types(history_wu, warmup_output_dir)
                    analyze_demographics(history_wu, warmup_output_dir)
                    analyze_waiting_times_top_10_percent(history_wu, warmup_output_dir)
                    analyze_waiting_times_top10_by_compatibility(history_wu, warmup_output_dir)
                    analyze_waiting_times(history_wu, warmup_output_dir)
                    print(f"Saved 'After Warm-Up' analysis to: {warmup_output_dir}")
                else:
                    print(" [SKIP] No data left after warm-up filtering.")

                # LAST 20 EPOCHS
                max_epoch = results_df["Epoch"].max()
                cutoff_epoch = max_epoch - 20
                print(f"\n [EXTRA] Running Last 20 Epochs Analysis (Epochs > {cutoff_epoch})...")

                last_epochs_output_dir = os.path.join(output_dir, "last_20_epochs")
                if not os.path.exists(last_epochs_output_dir):
                    os.makedirs(last_epochs_output_dir)

                # Filter Data
                results_le = results_df[results_df["Epoch"] > cutoff_epoch].copy()
                
                if "Epoch" in history_df.columns:
                    history_le = history_df[history_df["Epoch"] > cutoff_epoch].copy()
                else:
                    history_le = history_df.copy() # Fallback

                if not results_le.empty and not history_le.empty:
                    analyze_performance(results_le, last_epochs_output_dir)
                    analyze_outcomes(history_le, last_epochs_output_dir)
                    analyze_match_types(history_le, last_epochs_output_dir)
                    analyze_demographics(history_le, last_epochs_output_dir)
                    analyze_waiting_times_top_10_percent(history_le, last_epochs_output_dir)
                    analyze_waiting_times_top10_by_compatibility(history_le, last_epochs_output_dir)
                    analyze_waiting_times(history_le, last_epochs_output_dir)
                    print(f"Saved 'Last 20 Epochs' analysis to: {last_epochs_output_dir}")
                else:
                    print(" [SKIP] No data left after last 20 epochs filtering.")
                
                # FIRST 20 EPOCHS
                first_epochs = 20
                period = results_df["Period"].iloc[0] if "Period" in results_df.columns else 90
                cutoff_day = first_epochs * period
                print(f"\n [EXTRA] Running 'First 20 Epochs' Analysis (Epochs <= {first_epochs})...")

                first_epochs_output_dir = os.path.join(output_dir, "first_20_epochs")
                if not os.path.exists(first_epochs_output_dir):
                    os.makedirs(first_epochs_output_dir)

                # Filter Data
                results_fe = results_df[results_df["Epoch"] <= first_epochs].copy()

                if "Epoch" in history_df.columns and not history_df.empty:
                    # Include all pairs that ARRIVED up to the cutoff day
                    history_fe = history_df[history_df["Arrival"] <= cutoff_day].copy()
                    
                    # Ignore future events (matches, departures) that happen after the cutoff day for those pairs
                    future_mask = history_fe["Departure"] > cutoff_day
                    
                    if future_mask.any():
                        # Define status as "UNMATCHED_END" (they were in the pool at the end of this period)
                        history_fe.loc[future_mask, "Status"] = "UNMATCHED_END"
                        history_fe.loc[future_mask, "WaitTime"] = cutoff_day - history_fe.loc[future_mask, "Arrival"]
                        
                        # Clean information about future events that are not relevant for this period
                        history_fe.loc[future_mask, "MatchType"] = np.nan
                        history_fe.loc[future_mask, "MatchGroup"] = np.nan
                        history_fe.loc[future_mask, "Epoch"] = first_epochs

                else:
                    history_fe = pd.DataFrame()
                    
                    if "Epoch" in history_df.columns:
                        history_fe = history_df[history_df["Epoch"] <= first_epochs].copy()
                    else:
                        history_fe = history_df.copy() # Fallback

                if not results_fe.empty and not history_fe.empty:
                    analyze_performance(results_fe, first_epochs_output_dir)
                    analyze_outcomes(history_fe, first_epochs_output_dir)
                    analyze_match_types(history_fe, first_epochs_output_dir)
                    analyze_demographics(history_fe, first_epochs_output_dir)
                    analyze_waiting_times_top_10_percent(history_fe, first_epochs_output_dir)
                    analyze_waiting_times_top10_by_compatibility(history_fe, first_epochs_output_dir)
                    analyze_waiting_times(history_fe, first_epochs_output_dir)
                    print(f"Saved 'First 20 Epochs' analysis to: {first_epochs_output_dir}")
                else:
                    print(" [SKIP] No data left after first 20 epochs filtering.")            

    print(f"\nResults for {solver_name} processed successfully.")

if __name__ == "__main__":

    selection = None

    # Arguments Mode
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Analyze KEP Results")
        parser.add_argument("solver", choices=list(POLICY_CONFIGS.keys()) + ["all"], help="Which solver to analyze")
        args = parser.parse_args()
        selection = args.solver

    # Interactive Mode
    else:
        print("\n--- KEP Analysis Selector ---")
        options = list(POLICY_CONFIGS.keys()) + ["all"]

        for i, opt in enumerate(options, 1):
            name = "Analyze All Solvers" if opt == "all" else POLICY_CONFIGS[opt]["name"]
            print(f"{i}. {name} ({opt})")

        try:
            choice = int(input("\nSelect a number to analyze: "))
            if 1 <= choice <= len(options):
                selection = options[choice - 1]
            else:
                print("Invalid selection.")
                exit()
        except ValueError:
            print("Please enter a number.")
            exit()

    # Execution
    if selection == "all":
        for key in POLICY_CONFIGS.keys():
            run_analysis(key)
    else:
        run_analysis(selection)
