import argparse
import math
import os
import sys
import glob
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

POLICY_CONFIGS = {
    "hier_with_wt": {
        "folder": "kep_hier_with_wt",
        "name": "Hierarchical (WT)"
    },
    "hier_no_wt": {
        "folder": "kep_hier_no_wt",
        "name": "Hierarchical (No WT)"
    },
    "max_cardinality": {
        "folder": "kep_max_cardinality",
        "name": "Maximum Cardinality"
    },
    "static": {
        "folder": "kep_static",
        "name": "Static"
    }
}

ACCEPTANCE_RATES = [0, 25, 50, 75, 100]
YEARS = [6, 12, 24]

BASE_INPUT_DIR = "processed_results" # Folder where the data is
BASE_OUTPUT_DIR = "plot_results"     # Folder where the plots will be saved 

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_combined_data(policy_key, year, filename, sub_dir=None):
    """
    Read a specific file from ALL acceptance rate folders for a given policy and year.
    """
    config = POLICY_CONFIGS[policy_key]
    policy_folder = config["folder"]
    
    combined_list = []

    for acceptance in ACCEPTANCE_RATES:
        base_path = os.path.join(BASE_INPUT_DIR, f"{year}_year_simulation", policy_folder, f"{acceptance}_acceptance")
        
        if sub_dir:
            file_path = os.path.join(base_path, sub_dir, filename)
        else:
            file_path = os.path.join(base_path, filename)

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df["Acceptance_Rate"] = acceptance
                combined_list.append(df)
            except Exception as e:
                print(f"[Error] Could not read {file_path}: {e}")
        else:
            print(f"[Warn] File not found: {file_path}")
            pass

    if combined_list:
        return pd.concat(combined_list, ignore_index=True)
    else:
        return pd.DataFrame()

def load_all_policies_data(year, filename, sub_dir=None):
    """
    Loads the specified file from ALL configured policies.
    """
    all_data = []
    
    for key, config in POLICY_CONFIGS.items():

        if key == "static" and year == 24:
            continue

        df = load_combined_data(key, year, filename, sub_dir)
        
        if not df.empty:
            df["Solver"] = config["name"]
            df["SolverKey"] = key
            all_data.append(df)
            
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_outcomes_analysis(policy_key, year, sub_dir, output_dir):
    """
    Generates plots for outcomes analysis:
    1. Combined Spaghetti Plot (Matched & Timeout)
    2. Summary Comparison Plot (Mean ± 95% CI)
    """    
    # Plot Spaghetti
    df_inst = load_combined_data(policy_key, year, "outcomes_per_instance.csv", sub_dir)
    
    if not df_inst.empty:
        plt.figure(figsize=(10, 6))
        
        metrics_config = {
            "MATCHED": {"color": "#0072B2", "label": "Matched"},
            "TIMEOUT": {"color": "#D55E00", "label": "Timeout"}
        }
        
        plotted_something = False
        for metric, props in metrics_config.items():
            if metric in df_inst.columns:
                plotted_something = True
                
                pivot_data = df_inst.pivot(index="Acceptance_Rate", columns="Instance", values=metric)
                
                plt.plot(pivot_data.index, pivot_data.values, color=props["color"], alpha=0.1, linewidth=1)
                
                mean_values = pivot_data.mean(axis=1)
                plt.plot(mean_values.index, mean_values.values, color=props["color"], linewidth=3, label=f"Mean {props['label']}")

        if plotted_something:
            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Number of Pairs")
            plt.xticks(ACCEPTANCE_RATES)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            
            filename = "outcomes_spaghetti_combined.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"-> Created Plot: {filename}")

    # Plot Mean ± 95% CI
    df_sum = load_combined_data(policy_key, year, "outcomes_summary.csv", sub_dir)
    
    if not df_sum.empty:
        df_plot = df_sum[df_sum["Metric"].isin(["MATCHED", "TIMEOUT"])].copy()
        
        if not df_plot.empty:
            plt.figure(figsize=(10, 6))
            
            colors = {"MATCHED": "#0072B2", "TIMEOUT": "#D55E00"}
            markers = {"MATCHED": "o", "TIMEOUT": "s"}
            
            for metric in ["MATCHED", "TIMEOUT"]:
                subset = df_plot[df_plot["Metric"] == metric].sort_values("Acceptance_Rate")
                if subset.empty: continue
                
                x = subset["Acceptance_Rate"]
                y = subset["Mean"]
                ci_lower = subset["CI_Lower_95"]
                ci_upper = subset["CI_Upper_95"]
                c = colors.get(metric, "black")
                
                plt.fill_between(x, ci_lower, ci_upper, color=c, alpha=0.15)
                plt.plot(x, y, color=c, marker=markers.get(metric, "o"), linewidth=2, label=f"{metric} (Mean ± 95% CI)")

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Number of Pairs")
            plt.xticks(ACCEPTANCE_RATES)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            
            filename = "outcomes_comparison_ci.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"      -> Created Plot: {filename}")

def plot_demographics_analysis(policy_key, year, sub_dir, output_dir):
    """
    Generates demographic plots.
    """

    # Plot Configurations
    demographics_configs = [
        {"suffix": "compatibility", "cols": ["Compat_Label"], "title": "Compatibility Type", "filter": None},
        {"suffix": "blood", "cols": ["BloodPatient"], "title": "Patient Blood Type", "filter": None},
        {"suffix": "pra", "cols": ["PRA"], "title": "PRA Level", "filter": None},
        {
            "suffix": "blood_and_compat", 
            "cols": ["BloodPatient", "Compat_Label"], 
            "title": "Blood O x Difficulty (Excl. Compatible)",
            "filter": lambda df: (df["BloodPatient"] == "O") & (df["Compat_Label"] != "Compatible")
        },
        {
            "suffix": "pra_and_compat", 
            "cols": ["PRA", "Compat_Label"], 
            "title": "High PRA x Difficulty (Excl. Compatible)",
            "filter": lambda df: (df["PRA"] == "High") & (df["Compat_Label"] != "Compatible")
        }
    ]

    for config in demographics_configs:
        suffix = config["suffix"]
        cols_names = config["cols"]
        filter_func = config["filter"]

        df_inst = load_combined_data(policy_key, year, f"demographics_{suffix}_per_instance.csv", sub_dir)
        df_sum = load_combined_data(policy_key, year, f"demographics_{suffix}_summary.csv", sub_dir)

        if df_inst.empty or df_sum.empty:
            continue

        # Apply filter
        if filter_func:
            try:
                df_inst = df_inst[filter_func(df_inst)].copy()
                df_sum = df_sum[filter_func(df_sum)].copy()
            except Exception:
                continue
            
        if df_inst.empty: continue

        if len(cols_names) > 1:
            df_inst["Plot_Category"] = df_inst[cols_names].astype(str).agg(" - ".join, axis=1)
            df_sum["Plot_Category"] = df_sum[cols_names].astype(str).agg(" - ".join, axis=1)
        else:
            df_inst["Plot_Category"] = df_inst[cols_names[0]]
            df_sum["Plot_Category"] = df_sum[cols_names[0]]

        categories = sorted(df_inst["Plot_Category"].dropna().unique())
        n_cats = len(categories)
        if n_cats == 0: continue

        if n_cats <= 3:
            nrows, ncols = 1, n_cats
        elif n_cats == 4:
            nrows, ncols = 2, 2
        else:
            ncols = 3
            nrows = math.ceil(n_cats / ncols)

        figsize = (5 * ncols, 4 * nrows)

        # A. SPAGHETTI PLOTS
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes_flat = axes.flatten() if n_cats > 1 else [axes]

        for idx, ax in enumerate(axes_flat):
            if idx >= n_cats:
                ax.axis("off")
                continue
            
            cat = categories[idx]
            subset_inst = df_inst[df_inst["Plot_Category"] == cat]
            
            pivot_matched = subset_inst.pivot(index="Acceptance_Rate", columns="Instance", values="Matched")
            pivot_total = subset_inst.pivot(index="Acceptance_Rate", columns="Instance", values="Total")
            
            has_immuno = False
            pivot_immuno = None
            if "Immuno" in subset_inst.columns:
                if subset_inst["Immuno"].sum() > 0:
                    has_immuno = True
                    pivot_immuno = subset_inst.pivot(index="Acceptance_Rate", columns="Instance", values="Immuno")

            
            # 1. Total Pool
            mean_total = pivot_total.mean(axis=1)
            ax.plot(mean_total.index, mean_total.values, color="black", linestyle="--", linewidth=2, label="Total Pairs")

            # 2. Matched
            ax.plot(pivot_matched.index, pivot_matched.values, color="#0072B2", alpha=0.1, linewidth=1)
            mean_matched = pivot_matched.mean(axis=1)
            ax.plot(mean_matched.index, mean_matched.values, color="#0072B2", linewidth=3, label="Matched")

            # 3. Immuno & Combined
            if has_immuno and pivot_immuno is not None:
                # A. Immuno
                ax.plot(pivot_immuno.index, pivot_immuno.values, color="#009E73", alpha=0.1, linewidth=1)
                mean_immuno = pivot_immuno.mean(axis=1)
                ax.plot(mean_immuno.index, mean_immuno.values, color="#009E73", linewidth=3, label="Immuno")
                
                # B. COMBINED: Matched + Immuno
                pivot_combined = pivot_matched + pivot_immuno
                mean_combined = pivot_combined.mean(axis=1)
                
                ax.plot(mean_combined.index, mean_combined.values, color="#CC79A7", linestyle="-.", linewidth=2.5, label="Match+Immuno")

            ax.set_title(cat, fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_xticks(ACCEPTANCE_RATES)
            
            if idx == 0:
                ax.legend(loc="upper left", fontsize="x-small", frameon=True)

        fig.text(0.5, 0.02, "Acceptance Rate (%)", ha="center", fontsize=12)
        fig.text(0.02, 0.5, "Count", va="center", rotation="vertical", fontsize=12)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
        
        filename = f"demographics_spaghetti_{suffix}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"-> Created Plot: {filename}")


        # B. CI PLOTS (Mean with CI)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes_flat = axes.flatten() if n_cats > 1 else [axes]

        for idx, ax in enumerate(axes_flat):
            if idx >= n_cats:
                ax.axis("off")
                continue
            
            cat = categories[idx]
            subset_sum = df_sum[df_sum["Plot_Category"] == cat].sort_values("Acceptance_Rate")
            if subset_sum.empty: continue

            x = subset_sum["Acceptance_Rate"]
            
            # 1. Total
            ax.plot(x, subset_sum["Total_Mean"], color="black", linestyle="--", marker="x", linewidth=1.5, label="Total Pairs")

            # 2. Matched
            ax.fill_between(x, subset_sum["Matched_CI_Lower"], subset_sum["Matched_CI_Upper"], color="#0072B2", alpha=0.2)
            ax.plot(x, subset_sum["Matched_Mean"], color="#0072B2", marker="o", linewidth=2, label="Matched")

            # 3. Immuno & Combined
            if "Immuno_Mean" in subset_sum.columns and subset_sum["Immuno_Mean"].sum() > 0:
                # A. Immuno
                ax.fill_between(x, subset_sum["Immuno_CI_Lower"], subset_sum["Immuno_CI_Upper"], color="#009E73", alpha=0.2)
                ax.plot(x, subset_sum["Immuno_Mean"], color="#009E73", marker="s", linewidth=2, label="Immuno")
                
                # B. COMBINED
                y_combined = subset_sum["Matched_Mean"] + subset_sum["Immuno_Mean"]
                ax.plot(x, y_combined, color="#CC79A7", linestyle="-.", linewidth=2.5, label="Match+Immuno")

            ax.set_title(cat, fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_xticks(ACCEPTANCE_RATES)
            
            if idx == 0:
                ax.legend(loc="best", fontsize="x-small", frameon=True)

        fig.text(0.5, 0.02, "Acceptance Rate (%)", ha="center", fontsize=12)
        fig.text(0.02, 0.5, "Count", va="center", rotation="vertical", fontsize=12)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
        
        filename = f"demographics_ci_{suffix}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"-> Created Plot: {filename}")

def plot_waiting_times_analysis(policy_key, year, sub_dir, output_dir):
    """
    Generates plots for waiting times analysis.
    """    
    # Waiting Times: Mean vs Median + CI
    df_sum = load_combined_data(policy_key, year, "waittime_general_summary.csv", sub_dir)
    
    if not df_sum.empty:
        plt.figure(figsize=(10, 6))
        
        x = df_sum["Acceptance_Rate"]
        
        if "Mean_Wait_CI_Lower" in df_sum.columns:
            plt.fill_between(x, df_sum["Mean_Wait_CI_Lower"], df_sum["Mean_Wait_CI_Upper"], color="#0072B2", alpha=0.15)
        
        plt.plot(x, df_sum["Mean_Wait_GlobalAvg"], label="Mean Wait Time", color="#0072B2", marker="o", linewidth=2.5)
        
        plt.plot(x, df_sum["Median_Wait_GlobalAvg"], label="Median Wait Time", color="#D55E00", marker="s", linestyle="--", linewidth=2.5)

        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel("Days")
        plt.xticks(ACCEPTANCE_RATES)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        filename = "waittime_general_mean_vs_median.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"      -> Created Plot: {filename}")

    # Spaghetti Plot
    df_inst = load_combined_data(policy_key, year, "waittime_general_per_instance.csv", sub_dir)
    
    if not df_inst.empty:
        plt.figure(figsize=(10, 6))
        
        pivot_data = df_inst.pivot(index="Acceptance_Rate", columns="Instance", values="Mean_Wait")
        
        plt.plot(pivot_data.index, pivot_data.values, color="#0072B2", alpha=0.1, linewidth=1)
        
        mean_vals = pivot_data.mean(axis=1)
        plt.plot(mean_vals.index, mean_vals.values, color="#0072B2", linewidth=3, label="Global Mean")

        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel("Average Wait Time (Days)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        filename = "waittime_spaghetti.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"      -> Created Plot: {filename}")

    # Demographics Plots
    demo_configs = [
        {
            "filename": "waittime_compatibility_summary.csv", 
            "col": "Compat_Label", 
            "title": "Compatibility",
            "filter": None,
            "out_suffix": "compatibility"
        },
        {
            "filename": "waittime_blood_and_compat_summary.csv", 
            "col": "BloodPatient", 
            "title": "Patient Blood (Incompatible Pairs)",
            "filter": ("Compat_Label", "Incompatible"),
            "out_suffix": "blood_incompatible"
        },
        {
            "filename": "waittime_pra_and_compat_summary.csv", 
            "col": "PRA", 
            "title": "PRA Level (Incompatible Pairs)",
            "filter": ("Compat_Label", "Incompatible"),
            "out_suffix": "pra_incompatible"
        }
    ]

    for conf in demo_configs:
        file_csv = conf["filename"]
        col_name = conf["col"]
        out_suffix = conf["out_suffix"]
        filter_rule = conf["filter"]

        df_demo_sum = load_combined_data(policy_key, year, file_csv, sub_dir)
        
        if df_demo_sum.empty: continue

        # Apply the filter if it is defined (Compat_Label == Incompatible)
        if filter_rule:
            filter_col, filter_val = filter_rule
            if filter_col in df_demo_sum.columns:
                df_demo_sum = df_demo_sum[df_demo_sum[filter_col] == filter_val]
            else:
                continue

        if df_demo_sum.empty: continue

        categories = sorted(df_demo_sum[col_name].dropna().unique())
        n_cats = len(categories)
        if n_cats == 0: continue

        cols = 3
        rows = math.ceil(n_cats / cols)
        figsize = (5 * cols, 4 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
        axes_flat = axes.flatten() if n_cats > 1 else [axes]

        for idx, ax in enumerate(axes_flat):
            if idx >= n_cats:
                ax.axis("off")
                continue
            
            cat = categories[idx]
            subset = df_demo_sum[df_demo_sum[col_name] == cat].sort_values("Acceptance_Rate")
            
            if subset.empty: continue

            x = subset["Acceptance_Rate"]
            y = subset["Mean_Wait_GlobalAvg"]
            
            ci_l = subset["Mean_Wait_CI_Lower"]
            ci_u = subset["Mean_Wait_CI_Upper"]

            ax.fill_between(x, ci_l, ci_u, color="#009E73", alpha=0.15)
            ax.plot(x, y, label="Mean Wait", color="#009E73", marker="o", linewidth=2)
            
            ax.set_title(cat, fontsize=12, fontweight="bold")
            ax.set_xticks(ACCEPTANCE_RATES)
            ax.grid(True, linestyle="--", alpha=0.6)
            
            if idx == 0:
                ax.set_ylabel("Days")
                ax.legend(fontsize="small")

        fig.text(0.5, 0.02, "Acceptance Rate (%)", ha="center", fontsize=12)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
        
        filename = f"waittime_demographics_{out_suffix}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"-> Created Plot: {filename}")

def plot_waiting_times_top10_analysis(policy_key, year, sub_dir, output_dir):
    """
    Generates plots specifically for the top 10% worst waiting times.
    """
    # Plot Mean + CI Worst 10% (Global)
    df_sum = load_combined_data(policy_key, year, "waittime_top10_global_summary.csv", sub_dir)
    
    if not df_sum.empty:
        plt.figure(figsize=(10, 6))
        
        x = df_sum["Acceptance_Rate"]
        
        if "Global_Mean_CI_Lower" in df_sum.columns and "Global_Mean_CI_Upper" in df_sum.columns:
            plt.fill_between(x, df_sum["Global_Mean_CI_Lower"], df_sum["Global_Mean_CI_Upper"], color="#800000", alpha=0.15)
        
        if "Global_Mean" in df_sum.columns:
            plt.plot(x, df_sum["Global_Mean"], label="Mean (Top 10%)", color="#800000", marker="o", linewidth=2.5)

        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel("Days (Avg of Top 10%)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        filename = "waittime_top10_global_mean.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"      -> Created Plot: {filename}")

    # Plot Spaghetti
    df_inst = load_combined_data(policy_key, year, "waittime_top10_global_per_instance.csv", sub_dir)
    
    if not df_inst.empty:
        plt.figure(figsize=(10, 6))
        
        if "Instance_Waiting_Time_Mean" in df_inst.columns:
            pivot_data = df_inst.pivot(index="Acceptance_Rate", columns="Instance", values="Instance_Waiting_Time_Mean")
            
            plt.plot(pivot_data.index, pivot_data.values, color="#800000", alpha=0.1, linewidth=1)
            
            mean_vals = pivot_data.mean(axis=1)
            plt.plot(mean_vals.index, mean_vals.values, color="#800000", linewidth=3, label="Global Mean (Top 10%)")

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Days")
            plt.xticks(ACCEPTANCE_RATES)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            
            filename = "waittime_top10_global_spaghetti.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
            print(f"-> Created Plot: {filename}")

    # Compatibility Plot for Top 10% Worst Waiting Times
    df_demo_sum = load_combined_data(policy_key, year, "waittime_top10_compatibility_summary.csv", sub_dir)
    
    if not df_demo_sum.empty and "Compatibility" in df_demo_sum.columns:
        col_name = "Compatibility"
        
        categories = sorted(df_demo_sum[col_name].dropna().unique())
        n_cats = len(categories)
        
        if n_cats > 0:
            cols = 3
            rows = math.ceil(n_cats / cols)
            figsize = (5 * cols, 4 * rows)

            fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
            axes_flat = axes.flatten() if n_cats > 1 else [axes]

            for idx, ax in enumerate(axes_flat):
                if idx >= n_cats:
                    ax.axis("off")
                    continue
                
                cat = categories[idx]
                subset = df_demo_sum[df_demo_sum[col_name] == cat].sort_values("Acceptance_Rate")
                
                if subset.empty: continue

                x = subset["Acceptance_Rate"]
                y = subset["Global_Mean"]
                
                if "Global_Mean_CI_Lower" in subset.columns and "Global_Mean_CI_Upper" in subset.columns:
                    ci_l = subset["Global_Mean_CI_Lower"]
                    ci_u = subset["Global_Mean_CI_Upper"]
                    ax.fill_between(x, ci_l, ci_u, color="#800000", alpha=0.15)

                ax.plot(x, y, label="Top 10% Mean", color="#800000", marker="o", linewidth=2)
                
                ax.set_title(cat, fontsize=12, fontweight="bold")
                ax.set_xticks(ACCEPTANCE_RATES)
                ax.grid(True, linestyle="--", alpha=0.6)
                
                if idx == 0:
                    ax.set_ylabel("Days")
                    ax.legend(fontsize="small")

            fig.text(0.5, 0.02, "Acceptance Rate (%)", ha="center", fontsize=12)
            plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
            
            filename = f"waittime_top10_demographics_compatibility.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
            print(f"-> Created Plot: {filename}")

def plot_pool_evolution(policy_key, year, sub_dir, output_dir):
    """
    Generates plots for pool evolution analysis.
    """
    df_sum_all = load_combined_data(policy_key, year, "pool_evolution_summary.csv", sub_dir)
    df_inst_all = load_combined_data(policy_key, year, "pool_evolution_per_instance.csv", sub_dir)

    if df_sum_all.empty:
        return

    metrics_config = {
        "Pool_Total": {"label": "Total Pool", "color": "black", "style": "-"},
        "Pool_Patient_O": {"label": "Patient O", "color": "#0072B2", "style": "-"},          # Azul
        "Pool_Patient_O_Donor_AB": {"label": "Patient O - Donor AB", "color": "#D55E00", "style": "-"}, # Vermelho
        "Timeouts_In_Epoch": {"label": "Timeouts (Per Epoch)", "color": "#009E73", "style": "--"} # Verde
    }

    if "Timeouts_In_Epoch_Mean" not in df_sum_all.columns:
        print("[WARN] 'Timeouts_In_Epoch' not found. Please re-run process_results.py.")
        return

    rates = sorted(df_sum_all["Acceptance_Rate"].unique())
    n_plots = len(rates)
    
    if n_plots == 0: return

    # Plot Mean ± CI
    fig_ci, axes_ci = plt.subplots(1, n_plots, figsize=(4 * n_plots, 6), sharey=True, sharex=True)
    if n_plots == 1: axes_ci = [axes_ci]

    for idx, acc in enumerate(rates):
        ax = axes_ci[idx]
        df_sum = df_sum_all[df_sum_all["Acceptance_Rate"] == acc].sort_values("Epoch")
        
        if df_sum.empty:
            ax.set_title(f"{acc}% (No Data)")
            continue

        x = df_sum["Epoch"]

        for metric_col, props in metrics_config.items():
            y_mean = df_sum[f"{metric_col}_Mean"]
            ci_lower = df_sum[f"{metric_col}_CI_Lower"]
            ci_upper = df_sum[f"{metric_col}_CI_Upper"]

            ax.fill_between(x, ci_lower, ci_upper, color=props["color"], alpha=0.1)
            ax.plot(x, y_mean, label=props["label"], color=props["color"], linestyle=props["style"], linewidth=2)

        ax.set_title(f"Acceptance: {acc}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Match Run (Epoch)")
        ax.grid(True, linestyle="--", alpha=0.6)
        
        if idx == 0:
            ax.set_ylabel("Count")
            ax.legend(loc="upper left", fontsize="small", frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    filename_ci = "pool_evolution_combined_ci.png"
    plt.savefig(os.path.join(output_dir, filename_ci), dpi=300)
    plt.close(fig_ci)
    print(f"      -> Created Plot: {filename_ci}")


    # Plot Spaghetti
    if df_inst_all.empty:
        return

    fig_sp, axes_sp = plt.subplots(1, n_plots, figsize=(4 * n_plots, 6), sharey=True, sharex=True)
    if n_plots == 1: axes_sp = [axes_sp]

    for idx, acc in enumerate(rates):
        ax = axes_sp[idx]
        df_inst = df_inst_all[df_inst_all["Acceptance_Rate"] == acc]
        
        if df_inst.empty:
            continue

        for metric_col, props in metrics_config.items():
            pivot_data = df_inst.pivot(index="Epoch", columns="Instance", values=metric_col)

            ax.plot(pivot_data.index, pivot_data.values, color=props["color"], alpha=0.05, linewidth=1)
            
            mean_vals = pivot_data.mean(axis=1)
            ax.plot(mean_vals.index, mean_vals.values, label=props["label"], color=props["color"], linestyle=props["style"], linewidth=3)

        ax.set_title(f"Acceptance: {acc}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Match Run (Epoch)")
        ax.grid(True, linestyle="--", alpha=0.6)
        
        if idx == 0:
            ax.set_ylabel("Count")
            ax.legend(loc="upper left", fontsize="small", frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    filename_sp = "pool_evolution_combined_spaghetti.png"
    plt.savefig(os.path.join(output_dir, filename_sp), dpi=300)
    plt.close(fig_sp)
    print(f"-> Created Plot: {filename_sp}")

def generate_plots(policy_key):
    if policy_key not in POLICY_CONFIGS:
        print(f"Solver '{policy_key}' not found.")
        return

    policy_name = POLICY_CONFIGS[policy_key]["name"]
    print(f"\n==========================================")
    print(f"Generating Plots for: {policy_name}")
    print(f"==========================================")

    for year in YEARS:
        if policy_key == "static" and year == 24: continue

        print(f"\n>>> Year Scenario: {year} Years <<<")
         
        scenarios = [
            (None, "Standard Analysis (Full Horizon)"),
            ("first_20_epochs", "First 20 Epochs"),
            ("last_20_epochs", "Last 20 Epochs")
        ]

        for sub_dir, scenario_name in scenarios:
            print(f" -> Plotting: {scenario_name}")

            # Create output directory
            output_folder_rel = f"{year}_year_simulation/{POLICY_CONFIGS[policy_key]['folder']}"
            if sub_dir:
                output_folder_rel = os.path.join(output_folder_rel, sub_dir)
            
            full_output_dir = os.path.join(BASE_OUTPUT_DIR, output_folder_rel)
            if not os.path.exists(full_output_dir):
                os.makedirs(full_output_dir)

            plot_outcomes_analysis(policy_key, year, sub_dir, full_output_dir)
            plot_demographics_analysis(policy_key, year, sub_dir, full_output_dir)
            
            if policy_key != "static":
                plot_waiting_times_analysis(policy_key, year, sub_dir, full_output_dir)
                plot_waiting_times_top10_analysis(policy_key, year, sub_dir, full_output_dir)

            if policy_key != "static" or sub_dir in [None, "after_warm_up"]:
                plot_pool_evolution(policy_key, year, sub_dir, full_output_dir)

            print(f"(Plots saved to {full_output_dir})")
        
        base_output_path = os.path.join(BASE_OUTPUT_DIR, f"{year}_year_simulation/{POLICY_CONFIGS[policy_key]['folder']}")
        
        # First 20 epochs vs Last 20 epochs
        plot_first20_vs_last20_comparison(policy_key, year, base_output_path)

# =============================================================================
# COMPARISON: BETWEEN SOLVERS
# =============================================================================

def plot_comparison_outcomes(year, sub_dir, output_dir):
    """
    Compares MATCHED, TIMEOUTS, and TOTAL_TRANSPLANTS (Matched + Immuno) across all policies.
    """
    print(" -> Plotting Comparison: Outcomes...")

    outcomes_output_dir = os.path.join(output_dir, "outcomes")
    if not os.path.exists(outcomes_output_dir):
        os.makedirs(outcomes_output_dir)

    df_sum = load_all_policies_data(year, "outcomes_summary.csv", sub_dir)
    df_inst = load_all_policies_data(year, "outcomes_per_instance.csv", sub_dir)
    
    if df_sum.empty: 
        return

    # Add TOTAL_TRANSPLANTS (MATCHED + IMMUNO) metric to the data
    if not df_inst.empty and "MATCHED" in df_inst.columns and "IMMUNO" in df_inst.columns:
        df_inst["TOTAL_TRANSPLANTS"] = df_inst["MATCHED"] + df_inst["IMMUNO"]
        
        agg_tt = df_inst.groupby(["Solver", "SolverKey", "Acceptance_Rate"]).agg(
            Mean=("TOTAL_TRANSPLANTS", "mean"),
            Std_Dev=("TOTAL_TRANSPLANTS", "std"),
            N_Instances=("Instance", "count")
        ).reset_index()
        
        agg_tt["Metric"] = "TOTAL_TRANSPLANTS"
        
        se = agg_tt["Std_Dev"] / np.sqrt(agg_tt["N_Instances"])
        agg_tt["CI_Lower_95"] = agg_tt["Mean"] - (1.96 * se)
        agg_tt["CI_Upper_95"] = agg_tt["Mean"] + (1.96 * se)
        
        df_sum = pd.concat([df_sum, agg_tt], ignore_index=True)

    metrics = {
        "MATCHED": "Matched Pairs",
        "TIMEOUT": "Timeouts",
        "TOTAL_TRANSPLANTS": "Total Transplants (Match + Immuno)"
    }

    solvers = df_sum["Solver"].unique()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

    # Plot with only a line for the mean values
    for metric, ylabel in metrics.items():
        subset = df_sum[df_sum["Metric"] == metric]
        if subset.empty: continue

        plt.figure(figsize=(10, 6))
        
        sns.lineplot(
            data=subset, 
            x="Acceptance_Rate", 
            y="Mean", 
            hue="Solver", 
            style="Solver", 
            markers=solver_markers,
            dashes=False,
            linewidth=2.5,
            palette=solver_colors,
            hue_order=solvers,
            style_order=solvers
        )

        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel(f"Mean {ylabel}")
        plt.xticks(ACCEPTANCE_RATES)
        plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        filename_normal = f"compare_outcomes_{metric.lower()}_normal.png"
        plt.savefig(os.path.join(outcomes_output_dir, filename_normal), dpi=300)
        plt.close()
        print(f"      -> Created Plot: {filename_normal}")

    # Plot with Mean ± 95% CI
    for metric, ylabel in metrics.items():
        subset = df_sum[df_sum["Metric"] == metric]
        if subset.empty:
            continue

        plt.figure(figsize=(10, 6))

        for solver in solvers:
            data = subset[subset["Solver"] == solver].sort_values("Acceptance_Rate")
            if data.empty:
                continue

            x = data["Acceptance_Rate"]
            y = data["Mean"]
            ci_lower = data["CI_Lower_95"]
            ci_upper = data["CI_Upper_95"]

            color = solver_colors[solver]
            marker = solver_markers[solver]

            # shaded CI
            plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)

            # mean line
            plt.plot(
                x, y,
                color=color,
                marker=marker,
                linewidth=2,
                label=solver
            )

        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel(f"{ylabel} (Mean ± 95% CI)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        filename_ci = f"compare_outcomes_{metric.lower()}_ci.png"
        plt.savefig(os.path.join(outcomes_output_dir, filename_ci), dpi=300)
        plt.close()

        print(f"-> Created Plot: {filename_ci}")
    
    if df_inst.empty:
        print("[WARN] No per-instance data found for Spaghetti plots.")
        return

    # Plot with a line for the mean values + Spaghetti (lines for each instance)
    for metric, ylabel in metrics.items():
        if metric not in df_inst.columns: continue

        num_solvers = len(solvers)
        
        fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
        
        if num_solvers == 1:
            axes = [axes]
            
        for i, solver in enumerate(solvers):
            ax = axes[i]
            df_solver = df_inst[df_inst["Solver"] == solver]
            
            color = solver_colors[solver]
            marker = solver_markers.get(solver, "o")
            
            if df_solver.empty:
                ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                continue
                
            pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values=metric)
            
            ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
            
            mean_vals = pivot_data.mean(axis=1)
            ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
            
            ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
            ax.set_xlabel("Acceptance Rate (%)")
            ax.set_xticks(ACCEPTANCE_RATES)
            ax.grid(True, linestyle="--", alpha=0.6)
            
            if i == 0:
                ax.set_ylabel(ylabel)
        plt.tight_layout()

        filename_spag = f"compare_outcomes_{metric.lower()}_spaghetti.png"
        plt.savefig(os.path.join(outcomes_output_dir, filename_spag), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"-> Created Plot: {filename_spag}")

def plot_comparison_demographics(year, sub_dir, output_dir):
    """
    Compares performance for demographic groups: Blood Type O and High PRA.
    """
    print(" -> Plotting Comparison: Demographics...")

    demo_output_dir = os.path.join(output_dir, "demographics")
    if not os.path.exists(demo_output_dir):
        os.makedirs(demo_output_dir)
    
    # Blood Type O
    df_blood_sum = load_all_policies_data(year, "demographics_blood_summary.csv", sub_dir)
    df_blood_inst = load_all_policies_data(year, "demographics_blood_per_instance.csv", sub_dir)
    
    if not df_blood_sum.empty:
        subset_O_sum = df_blood_sum[df_blood_sum["BloodPatient"] == "O"]
        
        if not subset_O_sum.empty:
            solvers = subset_O_sum["Solver"].unique()
            palette = sns.color_palette("tab10", len(solvers))
            solver_colors = dict(zip(solvers, palette))
            solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

            # Normal Plot (Only Mean)
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=subset_O_sum, x="Acceptance_Rate", y="Matched_Mean", 
                hue="Solver", style="Solver", markers=solver_markers, dashes=False,
                linewidth=2.5, palette=solver_colors
            )
            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Mean Matched Pairs (Type O)")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_O_normal = "compare_demographics_blood_O_normal.png"
            plt.savefig(os.path.join(demo_output_dir, fname_O_normal), dpi=300)
            plt.close()
            print(f"-> Created Plot: {fname_O_normal}")

            # Plot with Mean ± 95% CI
            plt.figure(figsize=(10, 6))
            for solver in solvers:
                data = subset_O_sum[subset_O_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                x = data["Acceptance_Rate"]
                y = data["Matched_Mean"]
                ci_lower = data["Matched_CI_Lower"]
                ci_upper = data["Matched_CI_Upper"]
                
                color = solver_colors[solver]
                marker = solver_markers[solver]

                plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
                plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Matched Pairs (Mean ± 95% CI)")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_O_ci = "compare_demographics_blood_O_ci.png"
            plt.savefig(os.path.join(demo_output_dir, fname_O_ci), dpi=300)
            plt.close()
            print(f"-> Created Plot: {fname_O_ci}")

            # Spaghetti Plot (Lines for each instance)
            if not df_blood_inst.empty:
                subset_O_inst = df_blood_inst[df_blood_inst["BloodPatient"] == "O"]
                
                if not subset_O_inst.empty:
                    num_solvers = len(solvers)
                    fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
                    
                    if num_solvers == 1:
                        axes = [axes]
                        
                    for i, solver in enumerate(solvers):
                        ax = axes[i]
                        df_solver = subset_O_inst[subset_O_inst["Solver"] == solver]
                        
                        color = solver_colors[solver]
                        
                        if df_solver.empty:
                            ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                            continue
                            
                        pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values="Matched")
                        
                        ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
                        
                        mean_vals = pivot_data.mean(axis=1)
                        ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
                        
                        ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                        ax.set_xlabel("Acceptance Rate (%)")
                        ax.set_xticks(ACCEPTANCE_RATES)
                        ax.grid(True, linestyle="--", alpha=0.6)
                        
                        if i == 0:
                            ax.set_ylabel("Matched Pairs (Type O)")
                            
                    plt.tight_layout()
                    
                    fname_O_spag = "compare_demographics_blood_O_spaghetti.png"
                    plt.savefig(os.path.join(demo_output_dir, fname_O_spag), dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"-> Created Plot: {fname_O_spag}")

    # High PRA
    df_pra_sum = load_all_policies_data(year, "demographics_pra_summary.csv", sub_dir)
    df_pra_inst = load_all_policies_data(year, "demographics_pra_per_instance.csv", sub_dir)
    
    if not df_pra_sum.empty:
        subset_pra_sum = df_pra_sum[df_pra_sum["PRA"] == "High"]
        
        if not subset_pra_sum.empty:
            solvers = subset_pra_sum["Solver"].unique()
            palette = sns.color_palette("tab10", len(solvers))
            solver_colors = dict(zip(solvers, palette))
            solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

            # Normal Plot (Only Mean)
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=subset_pra_sum, x="Acceptance_Rate", y="Matched_Mean", 
                hue="Solver", style="Solver", markers=solver_markers, dashes=False,
                linewidth=2.5, palette=solver_colors
            )
            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Mean Matched Pairs (High PRA)")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_pra_normal = "compare_demographics_high_PRA_normal.png"
            plt.savefig(os.path.join(demo_output_dir, fname_pra_normal), dpi=300)
            plt.close()
            print(f"-> Created Plot: {fname_pra_normal}")

            # Plot with Mean ± 95% CI
            plt.figure(figsize=(10, 6))
            for solver in solvers:
                data = subset_pra_sum[subset_pra_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                x = data["Acceptance_Rate"]
                y = data["Matched_Mean"]
                ci_lower = data["Matched_CI_Lower"]
                ci_upper = data["Matched_CI_Upper"]
                
                color = solver_colors[solver]
                marker = solver_markers[solver]
                
                plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
                plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Matched Pairs (Mean ± 95% CI)")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_pra_ci = "compare_demographics_high_PRA_ci.png"
            plt.savefig(os.path.join(demo_output_dir, fname_pra_ci), dpi=300)
            plt.close()
            print(f"-> Created Plot: {fname_pra_ci}")

            # Spaghetti Plot (Lines for each instance)
            if not df_pra_inst.empty:
                subset_pra_inst = df_pra_inst[df_pra_inst["PRA"] == "High"]
                
                if not subset_pra_inst.empty:
                    num_solvers = len(solvers)
                    fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
                    
                    if num_solvers == 1:
                        axes = [axes]
                        
                    for i, solver in enumerate(solvers):
                        ax = axes[i]
                        df_solver = subset_pra_inst[subset_pra_inst["Solver"] == solver]
                        
                        color = solver_colors[solver]
                        
                        if df_solver.empty:
                            ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                            continue
                            
                        pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values="Matched")
                        
                        # Linhas individuais (Spaghetti) com a cor do solver
                        ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
                        
                        # Linha da Média destacada a preto
                        mean_vals = pivot_data.mean(axis=1)
                        ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
                        
                        ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                        ax.set_xlabel("Acceptance Rate (%)")
                        ax.set_xticks(ACCEPTANCE_RATES)
                        ax.grid(True, linestyle="--", alpha=0.6)
                        
                        if i == 0:
                            ax.set_ylabel("Matched Pairs (High PRA)")
                            
                    plt.tight_layout()
                    
                    fname_pra_spag = "compare_demographics_high_PRA_spaghetti.png"
                    plt.savefig(os.path.join(demo_output_dir, fname_pra_spag), dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"-> Created Plot: {fname_pra_spag}")

def plot_comparison_compatibility(year, sub_dir, output_dir):
    """
    Compares the number of matches and total transplants by Compatibility Type.
    """
    print(" -> Plotting Comparison: Compatibility Types...")
    compatibility_output_dir = os.path.join(output_dir, "compatibility")
    if not os.path.exists(compatibility_output_dir):
        os.makedirs(compatibility_output_dir)

    df_sum = load_all_policies_data(year, "demographics_compatibility_summary.csv", sub_dir)
    df_inst = load_all_policies_data(year, "demographics_compatibility_per_instance.csv", sub_dir)
    
    if df_sum.empty or "Compat_Label" not in df_sum.columns: 
        return
        
    # Calculate Total Transplants (Matched + Immuno) for instance and summary data
    if not df_inst.empty and "Matched" in df_inst.columns and "Immuno" in df_inst.columns:
        df_inst["Total_Transplants"] = df_inst["Matched"] + df_inst["Immuno"]
        
        agg_tt = df_inst.groupby(["Solver", "Acceptance_Rate", "Compat_Label"]).agg(
            Total_Transplants_Mean=("Total_Transplants", "mean"),
            Std_Dev=("Total_Transplants", "std"),
            N_Instances=("Instance", "count")
        ).reset_index()
        
        se = agg_tt["Std_Dev"] / np.sqrt(agg_tt["N_Instances"])
        agg_tt["Total_Transplants_CI_Lower"] = agg_tt["Total_Transplants_Mean"] - (1.96 * se)
        agg_tt["Total_Transplants_CI_Upper"] = agg_tt["Total_Transplants_Mean"] + (1.96 * se)
        
        # Merge the new aggregate metrics back into the main summary dataframe
        df_sum = pd.merge(
            df_sum, 
            agg_tt[["Solver", "Acceptance_Rate", "Compat_Label", "Total_Transplants_Mean", "Total_Transplants_CI_Lower", "Total_Transplants_CI_Upper"]], 
            on=["Solver", "Acceptance_Rate", "Compat_Label"], 
            how="left"
        )

    compat_types = sorted(df_sum["Compat_Label"].dropna().unique())

    solvers = df_sum["Solver"].unique()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

    for c_type in compat_types:
        subset_sum = df_sum[df_sum["Compat_Label"] == c_type]
        if subset_sum.empty: continue

        safe_name = c_type.replace(" ", "_").replace("-", "_").lower()

        # "Total_Transplants" only if is "Half-Compatible"
        metrics = {"Matched": "Matched Pairs"}
        if c_type == "Half-Compatible":
            metrics["Total_Transplants"] = "Total Transplants (Match + Immuno)"

        for metric_col, metric_label in metrics.items():
            y_mean = f"{metric_col}_Mean"
            y_ci_lower = f"{metric_col}_CI_Lower"
            y_ci_upper = f"{metric_col}_CI_Upper"

            if y_mean not in subset_sum.columns:
                continue

            # Plot with only a line for the mean values
            plt.figure(figsize=(10, 6))
            
            sns.lineplot(
                data=subset_sum, 
                x="Acceptance_Rate", 
                y=y_mean, 
                hue="Solver", 
                style="Solver", 
                markers=solver_markers, 
                dashes=False,
                linewidth=2.5, 
                palette=solver_colors,
                hue_order=solvers,
                style_order=solvers
            )
            
            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel(f"Mean {metric_label}")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_normal = f"compare_compatibility_{safe_name}_{metric_col.lower()}_normal.png"
            plt.savefig(os.path.join(compatibility_output_dir, fname_normal), dpi=300)
            plt.close()
            print(f"      -> Created Plot: compatibility/{fname_normal}")

            # Plot with Mean ± 95% CI
            if y_ci_lower in subset_sum.columns and y_ci_upper in subset_sum.columns:
                plt.figure(figsize=(10, 6))
                for solver in solvers:
                    data = subset_sum[subset_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                    if data.empty: continue
                    
                    x = data["Acceptance_Rate"]
                    y = data[y_mean]
                    ci_lower = data[y_ci_lower]
                    ci_upper = data[y_ci_upper]
                    
                    color = solver_colors[solver]
                    marker = solver_markers[solver]
                    
                    plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
                    plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

                plt.xlabel("Acceptance Rate (%)")
                plt.ylabel(f"{metric_label} ({c_type} - Mean ± 95% CI)")
                plt.xticks(ACCEPTANCE_RATES)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                
                fname_ci = f"compare_compatibility_{safe_name}_{metric_col.lower()}_ci.png"
                plt.savefig(os.path.join(compatibility_output_dir, fname_ci), dpi=300)
                plt.close()
                print(f"      -> Created Plot: compatibility/{fname_ci}")

            # Plot with a line for the mean values + Spaghetti (lines for each instance)
            if not df_inst.empty and metric_col in df_inst.columns:
                subset_inst = df_inst[df_inst["Compat_Label"] == c_type]
                if not subset_inst.empty:
                    num_solvers = len(solvers)
                    
                    fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
                    
                    if num_solvers == 1:
                        axes = [axes]
                        
                    for i, solver in enumerate(solvers):
                        ax = axes[i]
                        df_solver = subset_inst[subset_inst["Solver"] == solver]
                        
                        color = solver_colors[solver]
                        
                        if df_solver.empty:
                            ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                            continue
                            
                        pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values=metric_col)
                        
                        ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)

                        mean_vals = pivot_data.mean(axis=1)
                        ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)

                        ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                        ax.set_xlabel("Acceptance Rate (%)")
                        ax.set_xticks(ACCEPTANCE_RATES)
                        ax.grid(True, linestyle="--", alpha=0.6)

                        if i == 0:
                            ax.set_ylabel(f"{metric_label} ({c_type})")

                    plt.tight_layout()
                    
                    fname_spag = f"compare_compatibility_{safe_name}_{metric_col.lower()}_spaghetti.png"
                    plt.savefig(os.path.join(compatibility_output_dir, fname_spag), dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"      -> Created Plot: compatibility/{fname_spag}")

def plot_comparison_detailed_compatibility(year, sub_dir, output_dir):
    """
    Compares the distribution of Compatibility Types specifically for:
    1. Blood Type O Patients
    2. High PRA Patients
    """
    print(" -> Plotting Comparison: Detailed Compatibility (O & High PRA)...")
    
    det_comp_output_dir = os.path.join(output_dir, "detailed_compatibility")
    if not os.path.exists(det_comp_output_dir):
        os.makedirs(det_comp_output_dir)

    scenarios = [
        {"sum_file": "demographics_blood_and_compat_summary.csv", 
         "inst_file": "demographics_blood_and_compat_per_instance.csv", 
         "filter_col": "BloodPatient", "filter_val": "O", "title": "Blood Type O", "filename": "blood_O"},
        
        {"sum_file": "demographics_pra_and_compat_summary.csv", 
         "inst_file": "demographics_pra_and_compat_per_instance.csv", 
         "filter_col": "PRA", "filter_val": "High", "title": "High PRA", "filename": "pra_High"}
    ]

    for sc in scenarios:
        df_sum = load_all_policies_data(year, sc["sum_file"], sub_dir)
        df_inst = load_all_policies_data(year, sc["inst_file"], sub_dir)
        
        if df_sum.empty: continue

        if sc["filter_col"] in df_sum.columns:
            df_filtered_sum = df_sum[df_sum[sc["filter_col"]] == sc["filter_val"]]
        else:
            continue
            
        if df_filtered_sum.empty: continue

        compat_types = sorted(df_filtered_sum["Compat_Label"].dropna().unique())
        
        for c_type in compat_types:
            subset_sum = df_filtered_sum[df_filtered_sum["Compat_Label"] == c_type]
            if subset_sum.empty: continue

            solvers = subset_sum["Solver"].unique()
            palette = sns.color_palette("tab10", len(solvers))
            solver_colors = dict(zip(solvers, palette))
            solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))
            
            safe_ctype = c_type.replace(" ", "_").replace("-", "_").lower()

            # Plot with only a line for the mean values
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=subset_sum, x="Acceptance_Rate", y="Matched_Mean", 
                         hue="Solver", style="Solver", markers=solver_markers, dashes=False, 
                         linewidth=2.5, palette=solver_colors)
            
            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel(f"Mean Matches ({c_type})")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_normal = f"compare_detailed_compat_{sc['filename']}_{safe_ctype}_normal.png"
            plt.savefig(os.path.join(det_comp_output_dir, fname_normal), dpi=300)
            plt.close()

            # Plot with Mean ± 95% CI
            plt.figure(figsize=(10, 6))
            for solver in solvers:
                data = subset_sum[subset_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                x = data["Acceptance_Rate"]
                y = data["Matched_Mean"]
                ci_lower = data["Matched_CI_Lower"]
                ci_upper = data["Matched_CI_Upper"]
                
                color = solver_colors[solver]
                marker = solver_markers[solver]
                
                plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
                plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel(f"Mean Matches ({c_type}) ± 95% CI")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_ci = f"compare_detailed_compat_{sc['filename']}_{safe_ctype}_ci.png"
            plt.savefig(os.path.join(det_comp_output_dir, fname_ci), dpi=300)
            plt.close()

            # Plot with a line for the mean values + Spaghetti (lines for each instance)
            if not df_inst.empty and sc["filter_col"] in df_inst.columns:
                df_filtered_inst = df_inst[df_inst[sc["filter_col"]] == sc["filter_val"]]
                subset_inst = df_filtered_inst[df_filtered_inst["Compat_Label"] == c_type]
                
                if not subset_inst.empty:
                    num_solvers = len(solvers)
                    fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
                    
                    if num_solvers == 1:
                        axes = [axes]
                        
                    for i, solver in enumerate(solvers):
                        ax = axes[i]
                        df_solver = subset_inst[subset_inst["Solver"] == solver]
                        
                        color = solver_colors[solver]
                        
                        if df_solver.empty:
                            ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                            continue
                            
                        pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values="Matched")
                        
                        ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
                        
                        mean_vals = pivot_data.mean(axis=1)
                        ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
                        
                        ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                        ax.set_xlabel("Acceptance Rate (%)")
                        ax.set_xticks(ACCEPTANCE_RATES)
                        ax.grid(True, linestyle="--", alpha=0.6)
                        
                        if i == 0:
                            ax.set_ylabel(f"Matches ({c_type})")
                            
                    plt.tight_layout()
                    
                    fname_spag = f"compare_detailed_compat_{sc['filename']}_{safe_ctype}_spaghetti.png"
                    plt.savefig(os.path.join(det_comp_output_dir, fname_spag), dpi=300, bbox_inches="tight")
                    plt.close()

def plot_comparison_timeouts_incompatible(year, sub_dir, output_dir):
    """
    Compares the number of TIMEOUTS specifically for pairs INCOMPATIBLE.
    """
    print(" -> Plotting Comparison: Timeouts (Incompatible Pairs)...")
    timeouts_output_dir = os.path.join(output_dir, "timeouts")
    if not os.path.exists(timeouts_output_dir):
        os.makedirs(timeouts_output_dir)

    df_sum = load_all_policies_data(year, "demographics_compatibility_summary.csv", sub_dir)
    df_inst = load_all_policies_data(year, "demographics_compatibility_per_instance.csv", sub_dir)
    
    if df_sum.empty or "Compat_Label" not in df_sum.columns:
        print("   [WARN] No demographics compatibility data found.")
        return

    # Filter for "Incompatible"
    subset_sum = df_sum[df_sum["Compat_Label"] == "Incompatible"]
    
    if subset_sum.empty:
        print("[WARN] No data for 'Incompatible' label found.")
        return

    if "Timeout_Mean" not in subset_sum.columns:
        print("[WARN] 'Timeout_Mean' column missing. Please re-run process_results.py.")
        return

    solvers = subset_sum["Solver"].unique()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

    # Plot with only a line for the mean values
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=subset_sum, x="Acceptance_Rate", y="Timeout_Mean", 
        hue="Solver", style="Solver", markers=solver_markers, dashes=False,
        linewidth=2.5, palette=solver_colors
    )

    plt.xlabel("Acceptance Rate (%)")
    plt.ylabel("Mean Timeouts (Incompatible)")
    plt.xticks(ACCEPTANCE_RATES)
    plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    fname_normal = "compare_timeouts_incompatible_normal.png"
    plt.savefig(os.path.join(timeouts_output_dir, fname_normal), dpi=300)
    plt.close()
    print(f"      -> Created Plot: timeouts/{fname_normal}")

    # Plot with Mean ± 95% CI
    if "Timeout_CI_Lower" in subset_sum.columns:
        plt.figure(figsize=(10, 6))
        for solver in solvers:
            data = subset_sum[subset_sum["Solver"] == solver].sort_values("Acceptance_Rate")
            if data.empty: continue
            
            x = data["Acceptance_Rate"]
            y = data["Timeout_Mean"]
            ci_lower = data["Timeout_CI_Lower"]
            ci_upper = data["Timeout_CI_Upper"]
            
            color = solver_colors[solver]
            marker = solver_markers[solver]
            
            plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
            plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel("Timeouts (Mean ± 95% CI)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        
        fname_ci = "compare_timeouts_incompatible_ci.png"
        plt.savefig(os.path.join(timeouts_output_dir, fname_ci), dpi=300)
        plt.close()
        print(f"-> Created Plot: timeouts/{fname_ci}")

    # Plot with a line for the mean values + Spaghetti (lines for each instance)
    if not df_inst.empty and "Compat_Label" in df_inst.columns:
            subset_inst = df_inst[df_inst["Compat_Label"] == "Incompatible"]
            
            if not subset_inst.empty and "Timeout" in subset_inst.columns:
                num_solvers = len(solvers)
                fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
                
                if num_solvers == 1:
                    axes = [axes]
                    
                for i, solver in enumerate(solvers):
                    ax = axes[i]
                    df_solver = subset_inst[subset_inst["Solver"] == solver]
                    
                    color = solver_colors[solver]
                    
                    if df_solver.empty:
                        ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                        continue
                        
                    pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values="Timeout")
                    
                    ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
                    
                    mean_vals = pivot_data.mean(axis=1)
                    ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
                    
                    ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                    ax.set_xlabel("Acceptance Rate (%)")
                    ax.set_xticks(ACCEPTANCE_RATES)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    
                    if i == 0:
                        ax.set_ylabel("Timeouts (Incompatible Pairs)")
                        
                plt.tight_layout()
                
                fname_spag = "compare_timeouts_incompatible_spaghetti.png"
                plt.savefig(os.path.join(timeouts_output_dir, fname_spag), dpi=300, bbox_inches="tight")
                plt.close()
                print(f"-> Created Plot: timeouts/{fname_spag}")

def plot_comparison_match_structure(year, sub_dir, output_dir):
    """
    Compares the structure of matches (Cycles vs Chains) and sizes (2-way, 3-way, Chains).
    """
    print(" -> Plotting Comparison: Match Structure (Cycles vs Chains Combined)...")
    
    structure_dir = os.path.join(output_dir, "match_structure")
    if not os.path.exists(structure_dir):
        os.makedirs(structure_dir)

    df_sum = load_all_policies_data(year, "match_structure_summary.csv", sub_dir)
    df_inst_type = load_all_policies_data(year, "match_types_per_instance.csv", sub_dir)
    df_inst_size = load_all_policies_data(year, "match_sizes_per_instance.csv", sub_dir)

    if df_sum.empty: 
        print("[WARN] No match structure data found.")
        return

    solvers = df_sum["Solver"].unique()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    
    # Define grouped scenarios to plot
    scenarios = [
        {
            "name": "types_count",
            "title": "Match Types (Count of Pairs)",
            "ylabel": "Mean Pairs Matched",
            "is_pct": False,
            "metrics": ["Type_Cycle", "Type_Chain"],
            "labels": ["Cycles", "Chains"],
            "styles": ["solid", "dashed"],
            "df_inst": df_inst_type,
            "inst_cols": ["Cycle", "Chain"]
        },
        {
            "name": "types_fraction",
            "title": "Match Types (Percentage)",
            "ylabel": "Percentage of Total Matches (%)",
            "is_pct": True,
            "metrics": ["Type_Cycle_Fraction", "Type_Chain_Fraction"],
            "labels": ["Cycles (%)", "Chains (%)"],
            "styles": ["solid", "dashed"],
            "df_inst": df_inst_type,
            "inst_cols": ["Cycle_Fraction", "Chain_Fraction"]
        },
        {
            "name": "sizes_count",
            "title": "Match Sizes (Count of Pairs)",
            "ylabel": "Mean Pairs Matched",
            "is_pct": False,
            "metrics": ["Size_Pairs_in_2_Cycles", "Size_Pairs_in_3_Cycles", "Size_Pairs_in_Chains"],
            "labels": ["2-Way Cycles", "3-Way Cycles", "Chains"],
            "styles": ["solid", "dotted", "dashed"],
            "df_inst": df_inst_size,
            "inst_cols": ["Pairs_in_2_Cycles", "Pairs_in_3_Cycles", "Pairs_in_Chains"]
        }
    ]
    
    dash_sequences = {
        "solid": "",
        "dashed": (4, 1.5),
        "dotted": (1, 1)
    }

    for sc in scenarios:
        subset_sum = df_sum[df_sum["Metric"].isin(sc["metrics"])].copy()
        if subset_sum.empty: continue

        metric_to_label = dict(zip(sc["metrics"], sc["labels"]))
        subset_sum["Structure_Type"] = subset_sum["Metric"].map(metric_to_label)

        if sc["is_pct"]:
            subset_sum["Mean"] = subset_sum["Mean"] * 100
            if "CI_Lower" in subset_sum.columns:
                subset_sum["CI_Lower"] = subset_sum["CI_Lower"] * 100
                subset_sum["CI_Upper"] = subset_sum["CI_Upper"] * 100

        type_legend_elements = [
            mlines.Line2D([0], [0], color="black", lw=2.5, linestyle=sc["styles"][idx], label=sc["labels"][idx]) 
            for idx in range(len(sc["labels"]))
        ]
        solver_legend_elements = [
            mlines.Line2D([0], [0], color=solver_colors[s], lw=2.5, label=s) for s in solvers
        ]
        combined_legend = solver_legend_elements + type_legend_elements

        dashes_dict = {label: dash_sequences[style] for label, style in zip(sc["labels"], sc["styles"])}

        # Plot with only a line for the mean values
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=subset_sum, x="Acceptance_Rate", y="Mean",
            hue="Solver", style="Structure_Type",
            linewidth=2.5, palette=solver_colors, 
            hue_order=solvers, style_order=sc["labels"],
            dashes=dashes_dict
        )
        
        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel(sc["ylabel"])
        plt.xticks(ACCEPTANCE_RATES)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(handles=combined_legend, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        
        fname_normal = f"compare_structure_combined_{sc['name']}_normal.png"
        plt.savefig(os.path.join(structure_dir, fname_normal), dpi=300)
        plt.close()

        # Plot with Mean ± 95% CI
        if "CI_Lower" in subset_sum.columns and "CI_Upper" in subset_sum.columns:
            plt.figure(figsize=(12, 6))
            
            for solver in solvers:
                for idx, metric in enumerate(sc["metrics"]):
                    data = subset_sum[(subset_sum["Solver"] == solver) & (subset_sum["Metric"] == metric)].sort_values("Acceptance_Rate")
                    if data.empty: continue
                    
                    x = data["Acceptance_Rate"]
                    y = data["Mean"]
                    ci_l = data["CI_Lower"]
                    ci_u = data["CI_Upper"]
                    color = solver_colors[solver]
                    linestyle = sc["styles"][idx]
                    
                    plt.fill_between(x, ci_l, ci_u, color=color, alpha=0.10)
                    plt.plot(x, y, color=color, linewidth=2.5, linestyle=linestyle)

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel(f"{sc['ylabel']} (Mean ± 95% CI)")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(handles=combined_legend, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_ci = f"compare_structure_combined_{sc['name']}_ci.png"
            plt.savefig(os.path.join(structure_dir, fname_ci), dpi=300)
            plt.close()

        # Plot with a line for the mean values + Spaghetti (lines for each instance)
        inst_df = sc["df_inst"]
        
        if not inst_df.empty and all(col in inst_df.columns for col in sc["inst_cols"]):
            num_solvers = len(solvers)
            fig, axes = plt.subplots(1, num_solvers, figsize=(6 * num_solvers, 6), sharey=True)
            if num_solvers == 1: axes = [axes]

            for i, solver in enumerate(solvers):
                ax = axes[i]
                df_solver = inst_df[inst_df["Solver"] == solver].copy()

                if df_solver.empty:
                    ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                    continue

                for idx, inst_col in enumerate(sc["inst_cols"]):
                    if sc["is_pct"]:
                        df_solver[inst_col] = df_solver[inst_col] * 100
                        
                    color = solver_colors[solver]
                    linestyle = sc["styles"][idx]

                    pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values=inst_col)
                    
                    # Individual instances
                    ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.10, linestyle=linestyle, linewidth=1)
                    
                    # Mean line
                    mean_vals = pivot_data.mean(axis=1)
                    ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, linestyle=linestyle)

                ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                ax.set_xlabel("Acceptance Rate (%)")
                ax.set_xticks(ACCEPTANCE_RATES)
                ax.grid(True, linestyle="--", alpha=0.6)

                if i == 0:
                    ax.set_ylabel(sc["ylabel"])
                    ax.legend(handles=type_legend_elements, loc="best")

            plt.tight_layout()
            
            fname_spag = f"compare_structure_combined_{sc['name']}_spaghetti.png"
            plt.savefig(os.path.join(structure_dir, fname_spag), dpi=300, bbox_inches="tight")
            plt.close()
            
        print(f"-> Created Combined Plots for: {sc['title']}")

def plot_comparison_waittimes(year, sub_dir, output_dir):
    """
    Compares the Average and Median Waiting Time across ALL solvers.
    """
    print(" -> Plotting Comparison: Wait Times (General - Normal, CI, Spag)...")
    
    waittimes_output_dir = os.path.join(output_dir, "wt_general")
    if not os.path.exists(waittimes_output_dir):
        os.makedirs(waittimes_output_dir)

    df_sum = load_all_policies_data(year, "waittime_general_summary.csv", sub_dir)
    df_inst = load_all_policies_data(year, "waittime_general_per_instance.csv", sub_dir)
    
    if df_sum.empty: return

    solvers = df_sum["Solver"].unique()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

    # Plot with only a line for the mean values (Average and Median)
    metrics = {
        "Mean_Wait_GlobalAvg": "Average Wait Time (Days)",
        "Median_Wait_GlobalAvg": "Median Wait Time (Days)"
    }

    for col, ylabel in metrics.items():
        if col not in df_sum.columns: continue

        plt.figure(figsize=(10, 6))
        
        sns.lineplot(
            data=df_sum, 
            x="Acceptance_Rate", 
            y=col, 
            hue="Solver", 
            style="Solver", 
            markers=solver_markers, 
            dashes=False,
            linewidth=2.5,
            palette=solver_colors
        )

        metric_name = "Mean" if "Mean" in col else "Median"
        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel(ylabel)
        plt.xticks(ACCEPTANCE_RATES)
        plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        filename_normal = f"compare_waittime_{metric_name.lower()}_normal.png"
        plt.savefig(os.path.join(waittimes_output_dir, filename_normal), dpi=300)
        plt.close()
        print(f"-> Created Plot: waittimes/{filename_normal}")

    # Plot with Mean ± 95% CI
    if "Mean_Wait_CI_Lower" in df_sum.columns:
        plt.figure(figsize=(10, 6))
        
        for solver in solvers:
            data = df_sum[df_sum["Solver"] == solver].sort_values("Acceptance_Rate")
            if data.empty: continue
            
            x = data["Acceptance_Rate"]
            y = data["Mean_Wait_GlobalAvg"]
            ci_lower = data["Mean_Wait_CI_Lower"]
            ci_upper = data["Mean_Wait_CI_Upper"]
            
            color = solver_colors[solver]
            marker = solver_markers[solver]
            
            plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
            plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel("Average Wait Time (Days ± 95% CI)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        
        fname_ci = "compare_waittime_mean_ci.png"
        plt.savefig(os.path.join(waittimes_output_dir, fname_ci), dpi=300)
        plt.close()
        print(f"-> Created Plot: waittimes/{fname_ci}")

    # Plot with a line for the mean values + Spaghetti (lines for each instance)
    if not df_inst.empty and "Mean_Wait" in df_inst.columns:
        num_solvers = len(solvers)
        fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
        
        if num_solvers == 1:
            axes = [axes]
            
        for i, solver in enumerate(solvers):
            ax = axes[i]
            df_solver = df_inst[df_inst["Solver"] == solver]
            
            color = solver_colors[solver]
            
            if df_solver.empty:
                ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                continue
                
            pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values="Mean_Wait")
            
            ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
            
            mean_vals = pivot_data.mean(axis=1)
            ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
            ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
            ax.set_xlabel("Acceptance Rate (%)")
            ax.set_xticks(ACCEPTANCE_RATES)
            ax.grid(True, linestyle="--", alpha=0.6)
            
            if i == 0:
                ax.set_ylabel("Average Wait Time (Days)")
                
        plt.tight_layout()
        
        fname_spag = "compare_waittime_mean_spaghetti.png"
        plt.savefig(os.path.join(waittimes_output_dir, fname_spag), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"-> Created Plot: waittimes/{fname_spag}")

def plot_comparison_waittimes_compatibility(year, sub_dir, output_dir):
    """
    Compares the Average Wait Time by Compatibility Type across solvers.
    """
    print(" -> Plotting Comparison: Wait Times (Compatibility)...")
    
    wt_comp_output_dir = os.path.join(output_dir, "wt_compatibility")
    if not os.path.exists(wt_comp_output_dir):
        os.makedirs(wt_comp_output_dir)

    df_sum = load_all_policies_data(year, "waittime_compatibility_summary.csv", sub_dir)
    df_inst = load_all_policies_data(year, "waittime_compatibility_per_instance.csv", sub_dir)
    
    if df_sum.empty or "Compat_Label" not in df_sum.columns: 
        return

    compat_types = sorted(df_sum["Compat_Label"].dropna().unique())

    for c_type in compat_types:
        subset_sum = df_sum[df_sum["Compat_Label"] == c_type]
        if subset_sum.empty: continue

        solvers = subset_sum["Solver"].unique()
        palette = sns.color_palette("tab10", len(solvers))
        solver_colors = dict(zip(solvers, palette))
        solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

        safe_name = c_type.replace(" ", "_").replace("-", "_").lower()

        # Plot with only a line for the mean values
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=subset_sum, x="Acceptance_Rate", y="Mean_Wait_GlobalAvg", 
            hue="Solver", style="Solver", markers=solver_markers, dashes=False,
            linewidth=2.5, palette=solver_colors
        )
        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel(f"Average Wait Time (Days)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        
        fname_normal = f"compare_wt_compat_{safe_name}_normal.png"
        plt.savefig(os.path.join(wt_comp_output_dir, fname_normal), dpi=300)
        plt.close()

        # Plot with Mean ± 95% CI
        if "Mean_Wait_CI_Lower" in subset_sum.columns:
            plt.figure(figsize=(10, 6))
            for solver in solvers:
                data = subset_sum[subset_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                x = data["Acceptance_Rate"]
                y = data["Mean_Wait_GlobalAvg"]
                ci_lower = data["Mean_Wait_CI_Lower"]
                ci_upper = data["Mean_Wait_CI_Upper"]
                
                color = solver_colors[solver]
                marker = solver_markers[solver]
                
                plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
                plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Average Wait Time (Days ± 95% CI)")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_ci = f"compare_wt_compat_{safe_name}_ci.png"
            plt.savefig(os.path.join(wt_comp_output_dir, fname_ci), dpi=300)
            plt.close()

        # Plot with a line for the mean values + Spaghetti (lines for each instance)
        if not df_inst.empty and "Compat_Label" in df_inst.columns:
            subset_inst = df_inst[df_inst["Compat_Label"] == c_type]
            
            if not subset_inst.empty and "Mean_Wait" in subset_inst.columns:
                num_solvers = len(solvers)
                fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
                
                if num_solvers == 1:
                    axes = [axes]
                    
                for i, solver in enumerate(solvers):
                    ax = axes[i]
                    df_solver = subset_inst[subset_inst["Solver"] == solver]
                    
                    color = solver_colors[solver]
                    
                    if df_solver.empty:
                        ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                        continue
                        
                    pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values="Mean_Wait")
                    
                    ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
                    
                    mean_vals = pivot_data.mean(axis=1)
                    ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
                    
                    ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                    ax.set_xlabel("Acceptance Rate (%)")
                    ax.set_xticks(ACCEPTANCE_RATES)
                    ax.grid(True, linestyle="--", alpha=0.6)
                
                    if i == 0:
                        ax.set_ylabel("Average Wait Time (Days)")
                        
                plt.tight_layout()
                
                fname_spag = f"compare_wt_compat_{safe_name}_spaghetti.png"
                plt.savefig(os.path.join(wt_comp_output_dir, fname_spag), dpi=300, bbox_inches="tight")
                plt.close()
        
        print(f"-> Created Plots (Normal, CI, Spag) for: {c_type}")

def plot_comparison_waittimes_detailed_compatibility(year, sub_dir, output_dir):
    """
    Compares the Average Wait Time SPECIFICALLY for INCOMPATIBLE pairs,
    crossed with the two most difficult groups:
    1. Blood Type O Patients
    2. High PRA Patients
    """
    print(" -> Plotting Comparison: Wait Times Detailed (O & High PRA for Incompatible)...")
    
    wt_det_dir = os.path.join(output_dir, "waittimes_detailed_compatibility")
    if not os.path.exists(wt_det_dir):
        os.makedirs(wt_det_dir)
    
    scenarios = [
        {"sum_file": "waittime_blood_and_compat_summary.csv", 
         "inst_file": "waittime_blood_and_compat_per_instance.csv", 
         "filter_col": "BloodPatient", "filter_val": "O", "title": "Blood Type O", "filename": "blood_O"},
        
        {"sum_file": "waittime_pra_and_compat_summary.csv", 
         "inst_file": "waittime_pra_and_compat_per_instance.csv", 
         "filter_col": "PRA", "filter_val": "High", "title": "High PRA", "filename": "pra_High"}
    ]

    for sc in scenarios:
        df_sum = load_all_policies_data(year, sc["sum_file"], sub_dir)
        df_inst = load_all_policies_data(year, sc["inst_file"], sub_dir)
        
        if df_sum.empty: continue
        
        required_cols = [sc["filter_col"], "Compat_Label", "Mean_Wait_GlobalAvg"]
        if not all(col in df_sum.columns for col in required_cols):
            continue

        df_demographic_sum = df_sum[df_sum[sc["filter_col"]] == sc["filter_val"]]
        if df_demographic_sum.empty: continue

        # Only incompatible pairs
        subset_sum = df_demographic_sum[df_demographic_sum["Compat_Label"] == "Incompatible"]
        if subset_sum.empty: continue

        solvers = subset_sum["Solver"].unique()
        palette = sns.color_palette("tab10", len(solvers))
        solver_colors = dict(zip(solvers, palette))
        solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

        # Plot with only a line for the mean values
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=subset_sum, x="Acceptance_Rate", y="Mean_Wait_GlobalAvg", 
            hue="Solver", style="Solver", markers=solver_markers, dashes=False,
            linewidth=2.5, palette=solver_colors
        )
        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel(f"Average Wait Time (Days)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        fname_normal = f"compare_wt_incomp_{sc['filename']}_normal.png"
        plt.savefig(os.path.join(wt_det_dir, fname_normal), dpi=300)
        plt.close()

        # Plot with Mean ± 95% CI
        if "Mean_Wait_CI_Lower" in subset_sum.columns:
            plt.figure(figsize=(10, 6))
            for solver in solvers:
                data = subset_sum[subset_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                x = data["Acceptance_Rate"]
                y = data["Mean_Wait_GlobalAvg"]
                ci_lower = data["Mean_Wait_CI_Lower"]
                ci_upper = data["Mean_Wait_CI_Upper"]
                
                color = solver_colors[solver]
                marker = solver_markers[solver]
                
                plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
                plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Average Wait Time (Days ± 95% CI)")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_ci = f"compare_wt_incomp_{sc['filename']}_ci.png"
            plt.savefig(os.path.join(wt_det_dir, fname_ci), dpi=300)
            plt.close()

        # Plot with a line for the mean values + Spaghetti (lines for each instance)
        if not df_inst.empty and sc["filter_col"] in df_inst.columns and "Compat_Label" in df_inst.columns:

            df_demographic_inst = df_inst[df_inst[sc["filter_col"]] == sc["filter_val"]]
            subset_inst = df_demographic_inst[df_demographic_inst["Compat_Label"] == "Incompatible"]
            
            if not subset_inst.empty and "Mean_Wait" in subset_inst.columns:
                num_solvers = len(solvers)
                fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
                
                if num_solvers == 1:
                    axes = [axes]
                    
                for i, solver in enumerate(solvers):
                    ax = axes[i]
                    df_solver = subset_inst[subset_inst["Solver"] == solver]
                    
                    color = solver_colors[solver]
                    
                    if df_solver.empty:
                        ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                        continue
                        
                    pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values="Mean_Wait")
                    
                    ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
                    
                    mean_vals = pivot_data.mean(axis=1)
                    ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
                    
                    ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                    ax.set_xlabel("Acceptance Rate (%)")
                    ax.set_xticks(ACCEPTANCE_RATES)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    
                    if i == 0:
                        ax.set_ylabel("Average Wait Time (Days)")
                        
                plt.tight_layout()
                
                fname_spag = f"compare_wt_incomp_{sc['filename']}_spaghetti.png"
                plt.savefig(os.path.join(wt_det_dir, fname_spag), dpi=300, bbox_inches="tight")
                plt.close()
        
        print(f"-> Created Plots (Normal, CI, Spag) for: {sc['title']} (Incompatible Pairs Only)")

def plot_comparison_waittimes_top10_global(year, sub_dir, output_dir):
    """
    Compares the Average Wait Time for the Top 10% Worst Cases across solvers.
    """
    print(" -> Plotting Comparison: Wait Times Top 10% Global (Normal, CI, Spaghetti)...")
    
    wt_top10_output_dir = os.path.join(output_dir, "waittimes_top10")
    if not os.path.exists(wt_top10_output_dir):
        os.makedirs(wt_top10_output_dir)

    df_sum = load_all_policies_data(year, "waittime_top10_global_summary.csv", sub_dir)
    df_inst = load_all_policies_data(year, "waittime_top10_global_per_instance.csv", sub_dir)
    
    if df_sum.empty: return

    solvers = df_sum["Solver"].unique().tolist()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

    # Plot with only a line for the mean values
    if "Global_Mean" in df_sum.columns:
        plt.figure(figsize=(10, 6))
        
        sns.lineplot(
            data=df_sum, x="Acceptance_Rate", y="Global_Mean", 
            hue="Solver", style="Solver", markers=solver_markers, dashes=False,
            linewidth=2.5, palette=solver_colors
        )

        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel("Average Wait Time (Top 10% - Days)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        fname_normal = "compare_waittime_top10_global_mean_normal.png"
        plt.savefig(os.path.join(wt_top10_output_dir, fname_normal), dpi=300)
        plt.close()

    # Plot with Mean ± 95% CI
    if "Global_Mean_CI_Lower" in df_sum.columns and "Global_Mean_CI_Upper" in df_sum.columns:
        plt.figure(figsize=(10, 6))
        for solver in solvers:
            data = df_sum[df_sum["Solver"] == solver].sort_values("Acceptance_Rate")
            if data.empty: continue
            
            x = data["Acceptance_Rate"]
            y = data["Global_Mean"]
            ci_lower = data["Global_Mean_CI_Lower"]
            ci_upper = data["Global_Mean_CI_Upper"]
            
            color = solver_colors[solver]
            marker = solver_markers[solver]
            
            plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
            plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel("Wait Time Top 10% (Days ± 95% CI)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        
        fname_ci = "compare_waittime_top10_global_mean_ci.png"
        plt.savefig(os.path.join(wt_top10_output_dir, fname_ci), dpi=300)
        plt.close()

    # Plot with a line for the mean values + Spaghetti (lines for each instance)
    if not df_inst.empty and "Instance_Waiting_Time_Mean" in df_inst.columns:
        num_solvers = len(solvers)
        
        fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
        
        if num_solvers == 1:
            axes = [axes]
            
        for i, solver in enumerate(solvers):
            ax = axes[i]
            df_solver = df_inst[df_inst["Solver"] == solver]
            
            color = solver_colors[solver]
            marker = solver_markers[solver]
            
            if df_solver.empty:
                ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                continue
                
            pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values="Instance_Waiting_Time_Mean")
            
            ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
            
            mean_vals = pivot_data.mean(axis=1)
            ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
            
            ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
            ax.set_xlabel("Acceptance Rate (%)")
            ax.set_xticks(ACCEPTANCE_RATES)
            ax.grid(True, linestyle="--", alpha=0.6)
            
            if i == 0:
                ax.set_ylabel("Average Wait Time Top 10% (Days)")
        
        plt.tight_layout()
        
        fname_spag = "compare_waittime_top10_global_mean_spaghetti.png"
        plt.savefig(os.path.join(wt_top10_output_dir, fname_spag), dpi=300, bbox_inches="tight")
        plt.close()
        
    print("-> Top 10% Plots successfully created in waittimes_top10 folder.")

def plot_comparison_waittimes_top10_compatibility(year, sub_dir, output_dir):
    """
    Compares the Average Wait Time for the Top 10% Worst Cases, divided by Compatibility Type.
    """
    print(" -> Plotting Comparison: Wait Times Top 10% by Compatibility (Normal, CI, Spaghetti)...")
    
    wt_top10_comp_dir = os.path.join(output_dir, "waittimes_top10_compatibility")
    if not os.path.exists(wt_top10_comp_dir):
        os.makedirs(wt_top10_comp_dir)

    df_sum = load_all_policies_data(year, "waittime_top10_compatibility_summary.csv", sub_dir)
    df_inst = load_all_policies_data(year, "waittime_top10_compatibility_per_instance.csv", sub_dir)
    
    if df_sum.empty or "Compatibility" not in df_sum.columns: 
        return

    # Only analyze incompatible and half-compatible pairs
    target_types = ["Incompatible", "Half-Compatible"]
    all_types = df_sum["Compatibility"].dropna().unique()
    compat_types = [c for c in all_types if c in target_types]

    solvers = df_sum["Solver"].unique().tolist()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

    for c_type in compat_types:
        subset_sum = df_sum[df_sum["Compatibility"] == c_type]
        if subset_sum.empty: continue

        safe_name = c_type.replace(" ", "_").replace("-", "_").lower()

        y_col = "Global_Mean"

        # Plot with only a line for the mean values
        if y_col in subset_sum.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=subset_sum, x="Acceptance_Rate", y=y_col, 
                hue="Solver", style="Solver", markers=solver_markers, dashes=False,
                linewidth=2.5, palette=solver_colors,
                hue_order=solvers, style_order=solvers
            )
            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel(f"Avg Wait Time Top 10% ({c_type})")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_normal = f"compare_wt_top10_compat_{safe_name}_normal.png"
            plt.savefig(os.path.join(wt_top10_comp_dir, fname_normal), dpi=300)
            plt.close()

        # Plot with Mean ± 95% CI
        ci_lower_col = f"{y_col}_CI_Lower"
        ci_upper_col = f"{y_col}_CI_Upper"
        
        if ci_lower_col in subset_sum.columns and ci_upper_col in subset_sum.columns:
            plt.figure(figsize=(10, 6))
            for solver in solvers:
                data = subset_sum[subset_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                x = data["Acceptance_Rate"]
                y = data[y_col]
                ci_lower = data[ci_lower_col]
                ci_upper = data[ci_upper_col]
                
                color = solver_colors[solver]
                marker = solver_markers[solver]
                
                plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
                plt.plot(x, y, label=solver, color=color, marker=marker, linewidth=2.5)

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel(f"Wait Time Top 10% ({c_type}) ± 95% CI")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            fname_ci = f"compare_wt_top10_compat_{safe_name}_ci.png"
            plt.savefig(os.path.join(wt_top10_comp_dir, fname_ci), dpi=300)
            plt.close()

        # Plot with a line for the mean values + Spaghetti (lines for each instance)
        if not df_inst.empty and "Compatibility" in df_inst.columns:
            subset_inst = df_inst[df_inst["Compatibility"] == c_type]
            
            y_inst_col = "Instance_Mean"
            
            if not subset_inst.empty and y_inst_col in subset_inst.columns:
                num_solvers = len(solvers)
                fig, axes = plt.subplots(1, num_solvers, figsize=(5 * num_solvers, 6), sharey=True)
                
                if num_solvers == 1:
                    axes = [axes]
                    
                for i, solver in enumerate(solvers):
                    ax = axes[i]
                    df_solver = subset_inst[subset_inst["Solver"] == solver]
                    
                    color = solver_colors[solver]
                    marker = solver_markers[solver]
                    
                    if df_solver.empty:
                        ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                        continue
                        
                    pivot_data = df_solver.pivot(index="Acceptance_Rate", columns="Instance", values=y_inst_col)
                    
                    ax.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.15, linewidth=1)
                    
                    mean_vals = pivot_data.mean(axis=1)
                    ax.plot(mean_vals.index, mean_vals.values, color="black", linewidth=2.5, marker=None, markersize=6)
                    
                    ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                    ax.set_xlabel("Acceptance Rate (%)")
                    ax.set_xticks(ACCEPTANCE_RATES)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    
                    if i == 0:
                        ax.set_ylabel(f"Avg Wait Time Top 10% ({c_type})")
                
                plt.tight_layout()
                
                fname_spag = f"compare_wt_top10_compat_{safe_name}_spaghetti.png"
                plt.savefig(os.path.join(wt_top10_comp_dir, fname_spag), dpi=300, bbox_inches="tight")
                plt.close()
        
        print(f"-> Created Plots (Normal, CI, Spag) for: Top 10% {c_type}")

def plot_comparison_pool_evolution(year, sub_dir, output_dir):
    """
    Compares the evolution of Total Pool and Incompatible Pairs in a single graph,
    BUT SEPARATED FOR EACH ACCEPTANCE RATE.
    """
    print(" -> Plotting Comparison: Pool Evolution (Total & Incomp, by Acceptance Rate)...")

    pool_evo_dir = os.path.join(output_dir, "pool_evolution")
    if not os.path.exists(pool_evo_dir):
        os.makedirs(pool_evo_dir)

    df_sum = load_all_policies_data(year, "pool_evolution_summary.csv", sub_dir)
    df_inst = load_all_policies_data(year, "pool_evolution_per_instance.csv", sub_dir)
    
    if df_sum.empty or "Epoch" not in df_sum.columns or "Acceptance_Rate" not in df_sum.columns: 
        print("[WARN] No Pool Evolution data found or missing required columns.")
        return

    solvers = df_sum["Solver"].unique().tolist()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))

    acc_rates = sorted(df_sum["Acceptance_Rate"].dropna().unique())

    for ar in acc_rates:
        subset_sum = df_sum[df_sum["Acceptance_Rate"] == ar]
        if subset_sum.empty: continue

        safe_ar = int(ar)

        if "Pool_Total_Mean" not in subset_sum.columns or "Pool_Incompatible_Mean" not in subset_sum.columns:
            continue

        # Plot with only lines for the mean values (Total e Incompatible in the same graph)
        df_melt = subset_sum.melt(
            id_vars=["Epoch", "Solver"], 
            value_vars=["Pool_Total_Mean", "Pool_Incompatible_Mean"],
            var_name="Pool_Type", value_name="Mean_Size"
        )

        df_melt["Pool_Type"] = df_melt["Pool_Type"].map({
            "Pool_Total_Mean": "Overall Pool",
            "Pool_Incompatible_Mean": "Incompatible Pairs"
        })

        df_melt.rename(columns={"Solver": "Policy"}, inplace=True)

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df_melt, x="Epoch", y="Mean_Size", 
            hue="Policy", style="Pool_Type",
            hue_order=solvers, style_order=["Total Pool", "Incompatible Pool"],
            linewidth=2.5, palette=solver_colors, markers=False
        )
        
        plt.xlabel("Simulation Time (Periods)")
        plt.ylabel("Mean Active Pairs")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        fname_normal = f"compare_pool_evo_AR{safe_ar}_normal.png"
        plt.savefig(os.path.join(pool_evo_dir, fname_normal), dpi=300)
        plt.close()

        # Plot with Mean ± 95% CI
        if "Pool_Total_CI_Lower" in subset_sum.columns and "Pool_Incompatible_CI_Lower" in subset_sum.columns:
            plt.figure(figsize=(12, 6))
            
            for solver in solvers:
                data = subset_sum[subset_sum["Solver"] == solver].sort_values("Epoch")
                if data.empty: continue
                
                x = data["Epoch"]
                color = solver_colors[solver]
                
                y_tot = data["Pool_Total_Mean"]
                ci_l_tot = data["Pool_Total_CI_Lower"]
                ci_u_tot = data["Pool_Total_CI_Upper"]
                plt.fill_between(x, ci_l_tot, ci_u_tot, color=color, alpha=0.15)
                plt.plot(x, y_tot, color=color, linewidth=2.5, linestyle="solid")

                y_inc = data["Pool_Incompatible_Mean"]
                ci_l_inc = data["Pool_Incompatible_CI_Lower"]
                ci_u_inc = data["Pool_Incompatible_CI_Upper"]
                plt.fill_between(x, ci_l_inc, ci_u_inc, color=color, alpha=0.10)
                plt.plot(x, y_inc, color=color, linewidth=2.5, linestyle="dashed")

            plt.xlabel("Simulation Time (Epochs)")
            plt.ylabel("Active Pairs (Mean ± 95% CI)")
            
            import matplotlib.lines as mlines
            legend_elements = [mlines.Line2D([0], [0], color=solver_colors[s], lw=2.5, label=s) for s in solvers]
            legend_elements.append(mlines.Line2D([0], [0], color="black", lw=2.5, linestyle="solid", label="Total Pool"))
            legend_elements.append(mlines.Line2D([0], [0], color="black", lw=2.5, linestyle="dashed", label="Incompatible Pool"))
            
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            
            fname_ci = f"compare_pool_evo_AR{safe_ar}_ci.png"
            plt.savefig(os.path.join(pool_evo_dir, fname_ci), dpi=300)
            plt.close()

        # Plot with a line for the mean values + Spaghetti (lines for each instance)
        if not df_inst.empty and "Acceptance_Rate" in df_inst.columns:
            subset_inst = df_inst[df_inst["Acceptance_Rate"] == ar]
            
            if not subset_inst.empty and "Pool_Total" in subset_inst.columns and "Pool_Incompatible" in subset_inst.columns:
                num_solvers = len(solvers)
                fig, axes = plt.subplots(1, num_solvers, figsize=(6 * num_solvers, 6), sharey=True)
                
                if num_solvers == 1:
                    axes = [axes]
                    
                for i, solver in enumerate(solvers):
                    ax = axes[i]
                    df_solver = subset_inst[subset_inst["Solver"] == solver]
                    color = solver_colors[solver]
                    
                    if df_solver.empty:
                        ax.set_title(f"{solver} (No Data)", fontsize=12, fontweight="bold")
                        continue
                        
                    pivot_tot = df_solver.pivot(index="Epoch", columns="Instance", values="Pool_Total")
                    pivot_inc = df_solver.pivot(index="Epoch", columns="Instance", values="Pool_Incompatible")

                    ax.plot(pivot_tot.index, pivot_tot.values, color=color, alpha=0.10, linestyle="solid", linewidth=1)
                    ax.plot(pivot_inc.index, pivot_inc.values, color=color, alpha=0.10, linestyle="dashed", linewidth=1)
                    
                    mean_tot = pivot_tot.mean(axis=1)
                    mean_inc = pivot_inc.mean(axis=1)
                    
                    ax.plot(mean_tot.index, mean_tot.values, color="black", linewidth=2.5, linestyle="solid", label="Total Pool")
                    ax.plot(mean_inc.index, mean_inc.values, color="black", linewidth=2.5, linestyle="dashed", label="Incomp. Pool")
                    
                    ax.set_title(f"{solver}", fontsize=12, fontweight="bold", color=color)
                    ax.set_xlabel("Epochs")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    
                    if i == 0:
                        ax.set_ylabel("Active Pairs")
                        ax.legend(loc="upper left")
                
                plt.tight_layout()
                
                fname_spag = f"compare_pool_evo_AR{safe_ar}_spaghetti.png"
                plt.savefig(os.path.join(pool_evo_dir, fname_spag), dpi=300, bbox_inches="tight")
                plt.close()
        
        print(f"-> Created Combined Plots (Normal, CI, Spaghetti) for: Acceptance Rate {safe_ar}%")

def plot_comparison_histograms_incompatible(year, sub_dir, output_dir):
    """
    Generates comparative histograms focusing on INCOMPATIBLE pairs.
    View from 0 to 10 Years.
    Logarithmic Y-axis ONLY for "Resolved" metrics.
    """
    print(" -> Plotting Comparison Histograms (0-10 Years)...")

    hist_inc_dir = os.path.join(output_dir, "histograms_incompatible")
    if not os.path.exists(hist_inc_dir):
        os.makedirs(hist_inc_dir)
    
    df_all = load_all_policies_data(year, "histogram_outcomes_binned_incompatible.csv", sub_dir)
    
    if df_all.empty:
        print("   [WARN] No histogram data found.")
        return

    solvers = df_all["Solver"].unique()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = {solver: color for solver, color in zip(solvers, palette)}
    
    # Configuration for Limit
    MAX_SHOW_DAY = 10 * 360   # 3600 days (10 years)

    for acceptance in ACCEPTANCE_RATES:
        df_acc = df_all[df_all["Acceptance_Rate"] == acceptance].copy()
        if df_acc.empty: continue
        
        df_acc = df_acc.sort_values("Bin_Start_Day")

        # Ticks (0 to 10 Years)
        ticks_all = np.arange(0, MAX_SHOW_DAY + 1, 360)
        labels_all = [f"{int(y/360)}" for y in ticks_all]
        labels_all[0] = "0"

        def create_single_plot(metric_type, filename_prefix, title_suffix):
            plt.figure(figsize=(12, 6))
            
            for solver in solvers:
                data = df_acc[df_acc["Solver"] == solver]
                if data.empty: continue
                c = solver_colors[solver]

                # Filter data up to max show day
                data_plot = data[data["Bin_Start_Day"] <= MAX_SHOW_DAY]

                if metric_type == "Resolved":
                    # REMOVED the "label" argument here to prevent the clutter
                    plt.plot(data_plot["Bin_Start_Day"], data_plot["Count_Matched"], 
                             color=c, linestyle="-", linewidth=2)
                    plt.plot(data_plot["Bin_Start_Day"], data_plot["Count_Timeout"], 
                             color=c, linestyle="--", linewidth=2)
                else:
                    # Kept label for unmatched because it"s just a single line per solver
                    plt.plot(data_plot["Bin_Start_Day"], data_plot["Count_Unmatched"], 
                             color=c, linestyle="-", linewidth=2, label=solver)

            plt.xlabel("Years", fontsize=12)
            plt.xlim(0, MAX_SHOW_DAY)
            plt.xticks(ticks_all, labels_all)
            plt.grid(True, linestyle="--", alpha=0.6)
            
            if metric_type == "Resolved":
                plt.ylabel("Avg Pairs (Log Scale)", fontsize=12)
                plt.yscale("log")
                
                legend_elements = []
                
                # 1. Policy Group
                legend_elements.append(mlines.Line2D([], [], color="none", label="Policy"))
                for s in solvers:
                    legend_elements.append(mlines.Line2D([0], [0], color=solver_colors[s], lw=2.5, label=s))
                
                # 2. Outcome Type Group
                legend_elements.append(mlines.Line2D([], [], color="none", label="Outcome"))
                legend_elements.append(mlines.Line2D([0], [0], color="black", lw=2.5, linestyle="solid", label="Match"))
                legend_elements.append(mlines.Line2D([0], [0], color="black", lw=2.5, linestyle="dashed", label="Timeout"))
                
                plt.legend(handles=legend_elements, fontsize="small", loc="upper right")
            else:
                plt.ylabel("Avg Pairs", fontsize=12)
                plt.legend(fontsize="small", loc="upper right")
            
            plt.tight_layout()
            
            filename = f"incompatibles_{filename_prefix}_{acceptance}_acceptance.png"
            plt.savefig(os.path.join(hist_inc_dir, filename), dpi=300)
            plt.close()
            print(f"-> Created Plot: {filename}")

        create_single_plot("Resolved", "compare_hist_resolved", "Comparison: Matched & Timeout Pairs")
        
        if df_acc["Count_Unmatched"].sum() > 0:
            create_single_plot("Unmatched", "compare_hist_unmatched", "Comparison: Unmatched Pairs")

def plot_comparison_blood_O_leakage(year, sub_dir, output_dir):
    """
    Compares Blood O Leakage across policies.
    Generates 2 plots per scenario:
    - Normal (Mean + 95% CI)
    - Spaghetti (100 Instances + Mean)
    """
    print(" -> Plotting Comparison: Blood O Leakage (Normal & Spaghetti)...")

    leakage_dir = os.path.join(output_dir, "blood_o_leakage")
    if not os.path.exists(leakage_dir):
        os.makedirs(leakage_dir)
    
    file_sum = os.path.join("justification_results", "blood_O_flow_summary.csv")
    file_inst = os.path.join("justification_results", "blood_O_flow_per_instance.csv")
    
    df_sum = load_all_policies_data(year, file_sum, sub_dir)
    df_inst = load_all_policies_data(year, file_inst, sub_dir)

    if df_sum.empty: 
        print("[WARN] No Blood O flow summary found. Did you run process_results.py?")
        return

    if "DonorCompat" not in df_sum.columns:
        print("[ERROR] Column 'DonorCompat' missing in CSV.")
        return

    df_sum["DonorCompat"] = df_sum["DonorCompat"].astype(str)
    df_sum["RecipientCompat"] = df_sum["RecipientCompat"].astype(str)
    
    if not df_inst.empty:
        df_inst["DonorCompat"] = df_inst["DonorCompat"].astype(str)
        df_inst["RecipientCompat"] = df_inst["RecipientCompat"].astype(str)

    scenarios = [
        {
            "name": "Incompatible Donor to Non-Incompatible Recipient",
            "filter": lambda row: ("Incompatible" in row["DonorCompat"]) and ("Incompatible" not in row["RecipientCompat"])
        },
        {
            "name": "Half-Compatible Donor to Incompatible Recipient",
            "filter": lambda row: ("Half" in row["DonorCompat"]) and ("Incompatible" in row["RecipientCompat"])
        },
        {
            "name": "Non-Incompatible Donor to Incompatible Recipient",
            "filter": lambda row: ("Incompatible" not in row["DonorCompat"]) and ("Incompatible" in row["RecipientCompat"])
        },
        {
            "name": "Incompatible Donor to Half-Compatible Recipient",
            "filter": lambda row: ("Incompatible" in row["DonorCompat"]) and ("Half" in row["RecipientCompat"])
        }
    ]

    solvers = df_sum["Solver"].unique()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

    for sc in scenarios:
        print(f"[DEBUG] Processing scenario: {sc['name']}...")
        safe_name = sc["name"].replace(" ", "_").lower().replace("-", "_")
        
        # Mean + 95% CI
        mask_sum = df_sum.apply(sc["filter"], axis=1)
        subset_sum = df_sum[mask_sum].copy()

        if not subset_sum.empty:
            agg_sum = subset_sum.groupby(["Solver", "Acceptance_Rate"]).agg({
                "Mean_Count": "sum",
                "Std_Count": lambda x: np.sqrt((x**2).sum()), 
                "N_Instances": "mean"
            }).reset_index()

            z = 1.96
            agg_sum["Count_SE"] = agg_sum["Std_Count"] / np.sqrt(agg_sum["N_Instances"])
            agg_sum["Count_CI_Lower"] = (agg_sum["Mean_Count"] - (z * agg_sum["Count_SE"])).clip(lower=0)
            agg_sum["Count_CI_Upper"] = agg_sum["Mean_Count"] + (z * agg_sum["Count_SE"])

            plt.figure(figsize=(10, 6))
            
            for solver in solvers:
                data = agg_sum[agg_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                plt.plot(
                    data["Acceptance_Rate"], data["Mean_Count"], 
                    label=solver, color=solver_colors[solver], 
                    marker=solver_markers.get(solver, "o"), linewidth=2.5
                )
                
                if "Count_CI_Lower" in data.columns:
                    plt.fill_between(
                        data["Acceptance_Rate"], 
                        data["Count_CI_Lower"], data["Count_CI_Upper"], 
                        color=solver_colors[solver], alpha=0.15, edgecolor=None
                    )

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Mean Number of Donations")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Policy")
            plt.tight_layout()
            
            fname_normal = f"compare_blood_O_flow_custom_{safe_name}_normal.png"
            plt.savefig(os.path.join(leakage_dir, fname_normal), dpi=300)
            plt.close()
            print(f"-> Plot saved: {fname_normal}")
        else:
            print(f"[WARN] No data found for '{sc['name']}' in summary.")

        # Spaghetti plot with lines for each instance + Mean line
        if not df_inst.empty:
            mask_inst = df_inst.apply(sc["filter"], axis=1)
            subset_inst = df_inst[mask_inst].copy()
            
            if not subset_inst.empty:
                agg_inst = subset_inst.groupby(["Solver", "Acceptance_Rate", "Instance"])["Count"].sum().reset_index()

                plt.figure(figsize=(10, 6))
                
                for solver in solvers:
                    data_inst = agg_inst[agg_inst["Solver"] == solver]
                    if data_inst.empty: continue
                    
                    color = solver_colors[solver]
                    marker = solver_markers.get(solver, "o")

                    pivot_data = data_inst.pivot(index="Acceptance_Rate", columns="Instance", values="Count")
                    
                    plt.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.10, linewidth=1)

                    mean_vals = pivot_data.mean(axis=1)
                    plt.plot(mean_vals.index, mean_vals.values, color=color, linewidth=3.5, label=solver, marker=marker, markersize=6)

                plt.xlabel("Acceptance Rate (%)")
                plt.ylabel("Number of Donations")
                plt.xticks(ACCEPTANCE_RATES)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Policy")
                plt.tight_layout()
                
                fname_spag = f"compare_blood_O_flow_custom_{safe_name}_spaghetti.png"
                plt.savefig(os.path.join(leakage_dir, fname_spag), dpi=300)
                plt.close()
                print(f"-> Plot saved: {fname_spag}")
            else:
                print(f"[WARN] No instance data found for '{sc['name']}'.")

def plot_comparison_high_pra_incomp_sources(year, sub_dir, output_dir):
    """
    Compares the kidney sources for High PRA Incompatible recipients.
    """
    print(" -> Plotting Comparison: High PRA Incomp Sources (Normal & Spaghetti)...")

    sources_dir = os.path.join(output_dir, "high_pra_incomp_sources")
    if not os.path.exists(sources_dir):
        os.makedirs(sources_dir)
    
    file_sum = os.path.join("justification_results", "high_pra_incomp_kidney_sources_summary.csv")
    file_inst = os.path.join("justification_results", "high_pra_incomp_kidney_sources_per_instance.csv")
    
    df_sum = load_all_policies_data(year, file_sum, sub_dir)
    df_inst = load_all_policies_data(year, file_inst, sub_dir)

    if df_sum.empty: 
        print("[WARN] No High PRA incomp kidney sources summary found.")
        return

    if "DonorSource_Compat" not in df_sum.columns:
        print("[ERROR] Column 'DonorSource_Compat' missing in CSV.")
        return

    df_sum["DonorSource_Compat"] = df_sum["DonorSource_Compat"].astype(str)
    if not df_inst.empty:
        df_inst["DonorSource_Compat"] = df_inst["DonorSource_Compat"].astype(str)

    scenarios = [
        {
            "name": "Source: Altruist",
            "source_type": "Altruist"
        },
        {
            "name": "Source: Compatible",
            "source_type": "Compatible"
        },
        {
            "name": "Source: Half-Compatible",
            "source_type": "Half-Compatible"
        },
        {
            "name": "Source: Incompatible",
            "source_type": "Incompatible"
        }
    ]

    solvers = df_sum["Solver"].unique()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

    for sc in scenarios:
        print(f"[DEBUG] Processing scenario: {sc['name']}...")
        
        safe_name = sc["name"].replace(" ", "_").replace(":", "").lower().replace("-", "_")
        
        # Plot with Mean + CI
        subset_sum = df_sum[df_sum["DonorSource_Compat"] == sc["source_type"]].copy()

        if not subset_sum.empty:
            plt.figure(figsize=(10, 6))
            
            for solver in solvers:
                data = subset_sum[subset_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                plt.plot(
                    data["Acceptance_Rate"], data["Mean_Count"], 
                    label=solver, color=solver_colors[solver], 
                    marker=solver_markers.get(solver, "o"), linewidth=2.5
                )
                
                if "Count_CI_Lower" in data.columns:
                    plt.fill_between(
                        data["Acceptance_Rate"], 
                        data["Count_CI_Lower"], data["Count_CI_Upper"], 
                        color=solver_colors[solver], alpha=0.15, edgecolor=None
                    )

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Mean Number of Kidneys Received")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Policy")
            plt.tight_layout()
            
            fname_normal = f"compare_high_pra_sources_{safe_name}_normal.png"
            plt.savefig(os.path.join(sources_dir, fname_normal), dpi=300)
            plt.close()
            print(f"-> Plot saved: {fname_normal}")
        else:
            print(f"[WARN] No data found for '{sc['name']}' in summary.")

        # Plot Spaghetti
        if not df_inst.empty:
            subset_inst = df_inst[df_inst["DonorSource_Compat"] == sc["source_type"]].copy()
            
            if not subset_inst.empty:
                plt.figure(figsize=(10, 6))
                
                for solver in solvers:
                    data_inst = subset_inst[subset_inst["Solver"] == solver]
                    if data_inst.empty: continue
                    
                    color = solver_colors[solver]
                    marker = solver_markers.get(solver, "o")
                    
                    pivot_data = data_inst.pivot(index="Acceptance_Rate", columns="Instance", values="Count")
                    
                    plt.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.10, linewidth=1)
                    
                    mean_vals = pivot_data.mean(axis=1)
                    plt.plot(mean_vals.index, mean_vals.values, color=color, linewidth=3.5, label=solver, marker=marker, markersize=6)

                plt.xlabel("Acceptance Rate (%)")
                plt.ylabel("Number of Kidneys Received")
                plt.xticks(ACCEPTANCE_RATES)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Policy")
                plt.tight_layout()
                
                fname_spag = f"compare_high_pra_sources_{safe_name}_spaghetti.png"
                plt.savefig(os.path.join(sources_dir, fname_spag), dpi=300)
                plt.close()
                print(f"-> Plot saved: {fname_spag}")
            else:
                print(f"[WARN] No instance data found for '{sc['name']}'.")

def plot_comparison_incomp_kidney_sources(year, sub_dir, output_dir):
    """
    Compares the kidney sources for ALL Incompatible recipients.
    """
    print(" -> Plotting Comparison: Incomp Kidney Sources (Normal & Spaghetti)...")

    sources_dir = os.path.join(output_dir, "incomp_kidney_sources")
    if not os.path.exists(sources_dir):
        os.makedirs(sources_dir)
    
    file_sum = os.path.join("justification_results", "incompatible_kidney_sources_summary.csv")
    file_inst = os.path.join("justification_results", "incompatible_kidney_sources_per_instance.csv")
    
    df_sum = load_all_policies_data(year, file_sum, sub_dir)
    df_inst = load_all_policies_data(year, file_inst, sub_dir)

    if df_sum.empty: 
        print("   [WARN] No incomp kidney sources summary found.")
        return

    if "DonorSource_Compat" not in df_sum.columns:
        print("[ERROR] Column 'DonorSource_Compat' missing in CSV.")
        return

    df_sum["DonorSource_Compat"] = df_sum["DonorSource_Compat"].astype(str)
    if not df_inst.empty:
        df_inst["DonorSource_Compat"] = df_inst["DonorSource_Compat"].astype(str)

    scenarios = [
        {
            "name": "Source: Altruist",
            "source_type": "Altruist"
        },
        {
            "name": "Source: Compatible",
            "source_type": "Compatible"
        },
        {
            "name": "Source: Half-Compatible",
            "source_type": "Half-Compatible"
        },
        {
            "name": "Source: Incompatible",
            "source_type": "Incompatible"
        }
    ]

    solvers = df_sum["Solver"].unique()
    palette = sns.color_palette("tab10", len(solvers))
    solver_colors = dict(zip(solvers, palette))
    solver_markers = dict(zip(solvers, ["o", "s", "^", "D", "v", "<", ">"][:len(solvers)]))

    for sc in scenarios:
        print(f"[DEBUG] Processing scenario: {sc['name']}...")
        
        safe_name = sc["name"].replace(" ", "_").replace(":", "").lower().replace("-", "_")
        
        # Plot with Mean + CI
        subset_sum = df_sum[df_sum["DonorSource_Compat"] == sc["source_type"]].copy()

        if not subset_sum.empty:
            plt.figure(figsize=(10, 6))
            
            for solver in solvers:
                data = subset_sum[subset_sum["Solver"] == solver].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                plt.plot(
                    data["Acceptance_Rate"], data["Mean_Count"], 
                    label=solver, color=solver_colors[solver], 
                    marker=solver_markers.get(solver, "o"), linewidth=2.5
                )
                
                if "Count_CI_Lower" in data.columns:
                    plt.fill_between(
                        data["Acceptance_Rate"], 
                        data["Count_CI_Lower"], data["Count_CI_Upper"], 
                        color=solver_colors[solver], alpha=0.15, edgecolor=None
                    )

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Mean Number of Kidneys Received")
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Policy")
            plt.tight_layout()
            
            fname_normal = f"compare_incomp_sources_{safe_name}_normal.png"
            plt.savefig(os.path.join(sources_dir, fname_normal), dpi=300)
            plt.close()
            print(f"-> Plot saved: {fname_normal}")
        else:
            print(f"[WARN] No data found for '{sc['name']}' in summary.")

        # Plot Spaghetti
        if not df_inst.empty:
            subset_inst = df_inst[df_inst["DonorSource_Compat"] == sc["source_type"]].copy()
            
            if not subset_inst.empty:
                plt.figure(figsize=(10, 6))
                
                for solver in solvers:
                    data_inst = subset_inst[subset_inst["Solver"] == solver]
                    if data_inst.empty: continue
                    
                    color = solver_colors[solver]
                    marker = solver_markers.get(solver, "o")
                    
                    pivot_data = data_inst.pivot(index="Acceptance_Rate", columns="Instance", values="Count")
                    
                    plt.plot(pivot_data.index, pivot_data.values, color=color, alpha=0.10, linewidth=1)
                    
                    mean_vals = pivot_data.mean(axis=1)
                    plt.plot(mean_vals.index, mean_vals.values, color=color, linewidth=3.5, label=solver, marker=marker, markersize=6)

                plt.xlabel("Acceptance Rate (%)")
                plt.ylabel("Number of Kidneys Received")
                plt.xticks(ACCEPTANCE_RATES)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Policy")
                plt.tight_layout()
                
                fname_spag = f"compare_incomp_sources_{safe_name}_spaghetti.png"
                plt.savefig(os.path.join(sources_dir, fname_spag), dpi=300)
                plt.close()
                print(f"-> Plot saved: {fname_spag}")
            else:
                print(f"[WARN] No instance data found for '{sc['name']}'.")

def plot_cross_phase_all_policies_rates(year, dir1, name1, dir2, name2, output_dir, prefix):
    """
    Generates cross-phase comparison plots for all solvers using rates.
    """
    print(f" -> Plotting Cross-Phase Comparison: {name1} vs {name2}...")
    
    # 1. outcomes_summary.csv (MATCHED & TIMEOUTS)
    df_out1 = load_all_policies_data(year, "outcomes_summary.csv", dir1)
    df_out2 = load_all_policies_data(year, "outcomes_summary.csv", dir2)
    
    if not df_out1.empty and not df_out2.empty:
        df_out1["Phase"] = name1
        df_out2["Phase"] = name2
        df_out = pd.concat([df_out1, df_out2], ignore_index=True)
        df_out.rename(columns={"Solver": "Policy"}, inplace=True)
        
        for metric, ylabel in [("MATCHED_Rate", "Rate Matched Pairs (%)"), ("TIMEOUT_Rate", "Rate Timeouts (%)")]:
            subset = df_out[df_out["Metric"] == metric].copy()
            if subset.empty: continue

            subset["Mean"] = subset["Mean"] * 100
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=subset, x="Acceptance_Rate", y="Mean", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
            
            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel(ylabel)
            plt.xticks(ACCEPTANCE_RATES)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            
            filename = f"{prefix}_1_outcomes_{metric.replace('_Rate', '').lower()}_rate.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

    # 2. waittime_top10_global_summary.csv (Worst Case Wait Times)
    df_top1 = load_all_policies_data(year, "waittime_top10_global_summary.csv", dir1)
    df_top2 = load_all_policies_data(year, "waittime_top10_global_summary.csv", dir2)
    
    if not df_top1.empty and not df_top2.empty:
        df_top1["Phase"] = name1
        df_top2["Phase"] = name2
        df_top = pd.concat([df_top1, df_top2], ignore_index=True)
        df_top.rename(columns={"Solver": "Policy"}, inplace=True)
        
        if "Global_Mean" in df_top.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_top, x="Acceptance_Rate", y="Global_Mean", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Mean Wait Time (Top 10% Worst)")
            plt.xticks(ACCEPTANCE_RATES)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            
            filename = f"{prefix}_2_waittime_top10_global.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

    # 3. demographics_compatibility_summary.csv (Matches & Timeouts for INCOMPATIBLE)
    df_comp1 = load_all_policies_data(year, "demographics_compatibility_summary.csv", dir1)
    df_comp2 = load_all_policies_data(year, "demographics_compatibility_summary.csv", dir2)
    
    if not df_comp1.empty and not df_comp2.empty:
        df_comp1["Phase"] = name1
        df_comp2["Phase"] = name2
        df_comp = pd.concat([df_comp1, df_comp2], ignore_index=True)
        df_comp.rename(columns={"Solver": "Policy"}, inplace=True)
        
        if "Compat_Label" in df_comp.columns:
            subset = df_comp[df_comp["Compat_Label"] == "Incompatible"].copy()
            
            if not subset.empty:
                if "KEP_MatchRate_Mean" in subset.columns:
                    subset["KEP_MatchRate_Pct"] = subset["KEP_MatchRate_Mean"] * 100
                    
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=subset, x="Acceptance_Rate", y="KEP_MatchRate_Pct", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
                    plt.xlabel("Acceptance Rate (%)")
                    plt.ylabel("Match Rate (%)")
                    plt.xticks(ACCEPTANCE_RATES)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{prefix}_3_compat_incomp_match_rate.png"), dpi=300)
                    plt.close()

                if "Timeout_Mean" in subset.columns and "Total_Mean" in subset.columns:
                    subset["Timeout_Rate_Pct"] = (subset["Timeout_Mean"] / subset["Total_Mean"]) * 100
                    
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=subset, x="Acceptance_Rate", y="Timeout_Rate_Pct", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
                    plt.xlabel("Acceptance Rate (%)")
                    plt.ylabel("Timeout Rate (%)")
                    plt.xticks(ACCEPTANCE_RATES)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{prefix}_3_compat_incomp_timeout_rate.png"), dpi=300)
                    plt.close()

    # 4. waittime_top10_compatibility_summary.csv (Wait Times for INCOMPATIBLE Top 10%)
    df_wtcomp1 = load_all_policies_data(year, "waittime_top10_compatibility_summary.csv", dir1)
    df_wtcomp2 = load_all_policies_data(year, "waittime_top10_compatibility_summary.csv", dir2)
    
    if not df_wtcomp1.empty and not df_wtcomp2.empty:
        df_wtcomp1["Phase"] = name1
        df_wtcomp2["Phase"] = name2
        df_wtcomp = pd.concat([df_wtcomp1, df_wtcomp2], ignore_index=True)
        df_wtcomp.rename(columns={"Solver": "Policy"}, inplace=True)
        
        if "Compatibility" in df_wtcomp.columns and "Global_Mean" in df_wtcomp.columns:
            subset = df_wtcomp[df_wtcomp["Compatibility"] == "Incompatible"]
            
            if not subset.empty:
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=subset, x="Acceptance_Rate", y="Global_Mean", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
                
                plt.xlabel("Acceptance Rate (%)")
                plt.ylabel("Mean Wait Time (Days)")
                plt.xticks(ACCEPTANCE_RATES)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                
                filename = f"{prefix}_4_waittime_top10_incomp.png"
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.close()

    # 5. demographics_blood_and_compat_summary.csv (Matches for Blood O + Incompatible)
    df_bc1 = load_all_policies_data(year, "demographics_blood_and_compat_summary.csv", dir1)
    df_bc2 = load_all_policies_data(year, "demographics_blood_and_compat_summary.csv", dir2)
    
    if not df_bc1.empty and not df_bc2.empty:
        df_bc1["Phase"] = name1
        df_bc2["Phase"] = name2
        df_bc = pd.concat([df_bc1, df_bc2], ignore_index=True)
        df_bc.rename(columns={"Solver": "Policy"}, inplace=True)
        
        if "BloodPatient" in df_bc.columns and "Compat_Label" in df_bc.columns and "KEP_MatchRate_Mean" in df_bc.columns:
            subset = df_bc[(df_bc["BloodPatient"] == "O") & (df_bc["Compat_Label"] == "Incompatible")].copy()
            
            if not subset.empty:
                subset["KEP_MatchRate_Pct"] = subset["KEP_MatchRate_Mean"] * 100
                
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=subset, x="Acceptance_Rate", y="KEP_MatchRate_Pct", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
                
                plt.xlabel("Acceptance Rate (%)")
                plt.ylabel("Match Rate (%)")
                plt.xticks(ACCEPTANCE_RATES)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                
                filename = f"{prefix}_5_blood_O_incomp_match_rate.png"
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.close()

def plot_cross_phase_all_policies(year, dir1, name1, dir2, name2, output_dir, prefix):
    """
    Generate cross-phase comparison plots for all solvers using absolute numbers.
    """
    print(f" -> Plotting Cross-Phase Comparison: {name1} vs {name2}...")
    
    # 1. outcomes_summary.csv (MATCHED & TIMEOUTS)
    df_out1 = load_all_policies_data(year, "outcomes_summary.csv", dir1)
    df_out2 = load_all_policies_data(year, "outcomes_summary.csv", dir2)
    
    if not df_out1.empty and not df_out2.empty:
        df_out1["Phase"] = name1
        df_out2["Phase"] = name2
        df_out = pd.concat([df_out1, df_out2], ignore_index=True)
        df_out.rename(columns={"Solver": "Policy"}, inplace=True)
        
        for metric, ylabel in [("MATCHED", "Mean Matched Pairs"), ("TIMEOUT", "Mean Timeouts")]:
            subset = df_out[df_out["Metric"] == metric]
            if subset.empty: continue
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=subset, x="Acceptance_Rate", y="Mean", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
            
            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel(ylabel)
            plt.xticks(ACCEPTANCE_RATES)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            
            filename = f"{prefix}_1_outcomes_{metric.lower()}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

    # 2. waittime_top10_global_summary.csv (Worst Case Wait Times)
    df_top1 = load_all_policies_data(year, "waittime_top10_global_summary.csv", dir1)
    df_top2 = load_all_policies_data(year, "waittime_top10_global_summary.csv", dir2)
    
    if not df_top1.empty and not df_top2.empty:
        df_top1["Phase"] = name1
        df_top2["Phase"] = name2
        df_top = pd.concat([df_top1, df_top2], ignore_index=True)
        df_top.rename(columns={"Solver": "Policy"}, inplace=True)
        
        if "Global_Mean" in df_top.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_top, x="Acceptance_Rate", y="Global_Mean", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
            
            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel("Mean Wait Time (Top 10% Worst)")
            plt.xticks(ACCEPTANCE_RATES)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            
            filename = f"{prefix}_2_waittime_top10_global.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

    # 3. demographics_compatibility_summary.csv (Matches & Timeouts for INCOMPATIBLE)
    df_comp1 = load_all_policies_data(year, "demographics_compatibility_summary.csv", dir1)
    df_comp2 = load_all_policies_data(year, "demographics_compatibility_summary.csv", dir2)
    
    if not df_comp1.empty and not df_comp2.empty:
        df_comp1["Phase"] = name1
        df_comp2["Phase"] = name2
        df_comp = pd.concat([df_comp1, df_comp2], ignore_index=True)
        df_comp.rename(columns={"Solver": "Policy"}, inplace=True)
        
        if "Compat_Label" in df_comp.columns:
            subset = df_comp[df_comp["Compat_Label"] == "Incompatible"]
            
            if not subset.empty:
                if "Matched_Mean" in subset.columns:
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=subset, x="Acceptance_Rate", y="Matched_Mean", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
                    plt.xlabel("Acceptance Rate (%)")
                    plt.ylabel("Mean Matched Pairs (Incompatible)")
                    plt.xticks(ACCEPTANCE_RATES)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{prefix}_3_compat_incomp_matches.png"), dpi=300)
                    plt.close()

                if "Timeout_Mean" in subset.columns:
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=subset, x="Acceptance_Rate", y="Timeout_Mean", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
                    plt.xlabel("Acceptance Rate (%)")
                    plt.ylabel("Mean Timeouts (Incompatible)")
                    plt.xticks(ACCEPTANCE_RATES)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{prefix}_3_compat_incomp_timeouts.png"), dpi=300)
                    plt.close()

    # 4. waittime_top10_compatibility_summary.csv (Wait Times for INCOMPATIBLE Top 10%)
    df_wtcomp1 = load_all_policies_data(year, "waittime_top10_compatibility_summary.csv", dir1)
    df_wtcomp2 = load_all_policies_data(year, "waittime_top10_compatibility_summary.csv", dir2)
    
    if not df_wtcomp1.empty and not df_wtcomp2.empty:
        df_wtcomp1["Phase"] = name1
        df_wtcomp2["Phase"] = name2
        df_wtcomp = pd.concat([df_wtcomp1, df_wtcomp2], ignore_index=True)
        df_wtcomp.rename(columns={"Solver": "Policy"}, inplace=True)
        
        if "Compatibility" in df_wtcomp.columns and "Global_Mean" in df_wtcomp.columns:
            subset = df_wtcomp[df_wtcomp["Compatibility"] == "Incompatible"]
            
            if not subset.empty:
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=subset, x="Acceptance_Rate", y="Global_Mean", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
                
                plt.xlabel("Acceptance Rate (%)")
                plt.ylabel("Mean Wait Time (Days)")
                plt.xticks(ACCEPTANCE_RATES)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                
                filename = f"{prefix}_4_waittime_top10_incomp.png"
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.close()

    # 5. demographics_blood_and_compat_summary.csv (Matches for Blood O + Incompatible)
    df_bc1 = load_all_policies_data(year, "demographics_blood_and_compat_summary.csv", dir1)
    df_bc2 = load_all_policies_data(year, "demographics_blood_and_compat_summary.csv", dir2)
    
    if not df_bc1.empty and not df_bc2.empty:
        df_bc1["Phase"] = name1
        df_bc2["Phase"] = name2
        df_bc = pd.concat([df_bc1, df_bc2], ignore_index=True)
        df_bc.rename(columns={"Solver": "Policy"}, inplace=True)
        
        if "BloodPatient" in df_bc.columns and "Compat_Label" in df_bc.columns and "Matched_Mean" in df_bc.columns:
            subset = df_bc[(df_bc["BloodPatient"] == "O") & (df_bc["Compat_Label"] == "Incompatible")]
            
            if not subset.empty:
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=subset, x="Acceptance_Rate", y="Matched_Mean", hue="Policy", style="Phase", markers=True, linewidth=2.5, palette="tab10")
                
                plt.xlabel("Acceptance Rate (%)")
                plt.ylabel("Mean Matched Pairs")
                plt.xticks(ACCEPTANCE_RATES)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                
                filename = f"{prefix}_5_blood_O_incomp_matches.png"
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.close()

def generate_comparison_plots():
    print(f"\n==========================================")
    print(f"Generating SOLVER COMPARISON Plots")
    print(f"==========================================")

    for year in YEARS:
        
        out_folder_std = os.path.join(BASE_OUTPUT_DIR, "comparison", f"{year}_year", "standard")
        if not os.path.exists(out_folder_std):
            os.makedirs(out_folder_std)
            
        print(f"\n>>> Scenario: Standard ({year} Years) <<<")
        sub_dir = None  # None indicates the base "standard" folder
        
        plot_comparison_outcomes(year, sub_dir, out_folder_std)
        plot_comparison_demographics(year, sub_dir, out_folder_std)
        plot_comparison_compatibility(year, sub_dir, out_folder_std)
        plot_comparison_detailed_compatibility(year, sub_dir, out_folder_std)
        
        plot_comparison_timeouts_incompatible(year, sub_dir, out_folder_std)

        plot_comparison_match_structure(year, sub_dir, out_folder_std)

        plot_comparison_waittimes(year, sub_dir, out_folder_std)
        plot_comparison_waittimes_compatibility(year, sub_dir, out_folder_std)
        plot_comparison_waittimes_detailed_compatibility(year, sub_dir, out_folder_std)
        plot_comparison_waittimes_top10_global(year, sub_dir, out_folder_std)
        plot_comparison_waittimes_top10_compatibility(year, sub_dir, out_folder_std)
        
        plot_comparison_pool_evolution(year, sub_dir, out_folder_std)

        plot_comparison_histograms_incompatible(year, sub_dir, out_folder_std)

        plot_comparison_blood_O_leakage(year, sub_dir, out_folder_std)
        plot_comparison_high_pra_incomp_sources(year, sub_dir, out_folder_std)
        plot_comparison_incomp_kidney_sources(year, sub_dir, out_folder_std)
    
        print(f"    (Standard comparisons saved to {out_folder_std})")

        if year == 24:
            # FIRST 20 vs LAST 20 (Cross-Phase with All 3 Policies)
            out_folder_el = os.path.join(BASE_OUTPUT_DIR, "comparison", f"{year}_year", "early_vs_late")
            if not os.path.exists(out_folder_el):
                os.makedirs(out_folder_el)
                
            print(f"\n>>> Scenario: Early vs Late ({year} Years) <<<")
            
            plot_cross_phase_all_policies(
                year=year, 
                dir1="first_20_epochs", name1="First 20", 
                dir2="last_20_epochs",  name2="Last 20", 
                output_dir=out_folder_el, 
                prefix="early_vs_late"
            )

            plot_cross_phase_all_policies_rates(
                year=year, 
                dir1="first_20_epochs", name1="First 20", 
                dir2="last_20_epochs",  name2="Last 20", 
                output_dir=out_folder_el, 
                prefix="early_vs_late"
            )
            
            print(f"    (Early vs Late comparisons saved to {out_folder_el})")

# =============================================================================
# COMPARISON: FIRST 20 VS LAST 20 EPOCHS
# =============================================================================

def plot_first20_vs_last20_comparison(policy_key, year, output_dir):
    """
    Compares the first 20 epochs vs the last 20 epochs for a given solver and year, generating plots for outcomes and demographics.
    """
    policy_name = POLICY_CONFIGS[policy_key]["name"]
    print(f" -> Plotting First 20 vs Last 20 Comparison for {policy_name}...")

    comp_output_dir = os.path.join(output_dir, "comparison_first20_vs_last20")
    if not os.path.exists(comp_output_dir):
        os.makedirs(comp_output_dir)

    # Outcomes to compare: MATCHED, TIMEOUT, and their Rates
    df_first = load_combined_data(policy_key, year, "outcomes_summary.csv", sub_dir="first_20_epochs")
    df_last = load_combined_data(policy_key, year, "outcomes_summary.csv", sub_dir="last_20_epochs")

    if not df_first.empty and not df_last.empty:
        df_first["Phase"] = "First 20 Epochs"
        df_last["Phase"] = "Last 20 Epochs"
        df_combined = pd.concat([df_first, df_last], ignore_index=True)

        outcomes_config = [
            # Counts
            {"metric": "MATCHED", "title": "Matched Pairs (Count)", "is_rate": False, 
             "color_first": "#009E73", "color_last": "#D55E00"},
            {"metric": "TIMEOUT", "title": "Timeouts (Count)", "is_rate": False, 
             "color_first": "#56B4E9", "color_last": "#CC79A7"},
            # Rates
            {"metric": "MATCHED_Rate", "title": "Match Rate (%)", "is_rate": True, 
             "color_first": "#009E73", "color_last": "#D55E00"},
            {"metric": "TIMEOUT_Rate", "title": "Timeout Rate (%)", "is_rate": True, 
             "color_first": "#56B4E9", "color_last": "#CC79A7"}
        ]

        for conf in outcomes_config:
            metric = conf["metric"]
            is_rate = conf["is_rate"]
            subset = df_combined[df_combined["Metric"] == metric]
            
            if subset.empty: continue

            plt.figure(figsize=(10, 6))
            
            for phase, color in [("First 20 Epochs", conf["color_first"]), ("Last 20 Epochs", conf["color_last"])]:
                data = subset[subset["Phase"] == phase].sort_values("Acceptance_Rate")
                if data.empty: continue
                
                x = data["Acceptance_Rate"]
                y = data["Mean"] * 100 if is_rate else data["Mean"]
                
                ci_low = data["CI_Lower_95"] * 100 if is_rate else data["CI_Lower_95"]
                ci_high = data["CI_Upper_95"] * 100 if is_rate else data["CI_Upper_95"]

                plt.fill_between(x, ci_low, ci_high, color=color, alpha=0.15)
                plt.plot(x, y, label=phase, color=color, marker="o", linewidth=2.5)

            plt.xlabel("Acceptance Rate (%)")
            plt.ylabel(conf["title"])
            plt.xticks(ACCEPTANCE_RATES)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            
            filename = f"compare_early_late_{metric.lower()}.png"
            plt.savefig(os.path.join(comp_output_dir, filename), dpi=300)
            plt.close()
            print(f"-> Created Plot: {filename}")


    # Demographics to compare: Compatibility, Blood Type, PRA
    demo_configs = [
        {"suffix": "compatibility", "col": "Compat_Label", "title": "Compatibility"},
        {"suffix": "blood", "col": "BloodPatient", "title": "Blood Type"},
        {"suffix": "pra", "col": "PRA", "title": "PRA Level"}
    ]

    for conf in demo_configs:
        suffix = conf["suffix"]
        col_name = conf["col"]

        d_first = load_combined_data(policy_key, year, f"demographics_{suffix}_summary.csv", sub_dir="first_20_epochs")
        d_last = load_combined_data(policy_key, year, f"demographics_{suffix}_summary.csv", sub_dir="last_20_epochs")

        if d_first.empty or d_last.empty: continue

        d_first["Phase"] = "First 20 Epochs"
        d_last["Phase"] = "Last 20 Epochs"
        d_combined = pd.concat([d_first, d_last], ignore_index=True)

        categories = sorted(d_combined[col_name].dropna().unique())
        n_cats = len(categories)
        if n_cats == 0: continue

        cols = 3
        rows = math.ceil(n_cats / cols)
        
        plot_types = [
            {"col_y": "Matched_Mean", "ylabel": "Mean Count", "file_tag": "count", "is_rate": False},
            {"col_y": "KEP_MatchRate_Mean", "ylabel": "Match Rate (%)", "file_tag": "rate", "is_rate": True}
        ]

        for ptype in plot_types:
            figsize = (5 * cols, 4 * rows)
            fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
            axes_flat = axes.flatten() if n_cats > 1 else [axes]

            for idx, ax in enumerate(axes_flat):
                if idx >= n_cats:
                    ax.axis("off")
                    continue
                
                cat = categories[idx]
                subset = d_combined[d_combined[col_name] == cat]
                colors = {"First 20 Epochs": "#009E73", "Last 20 Epochs": "#D55E00"}

                for phase in ["First 20 Epochs", "Last 20 Epochs"]:
                    data = subset[subset["Phase"] == phase].sort_values("Acceptance_Rate")
                    if data.empty: continue
                    
                    y_val = data[ptype["col_y"]] * 100 if ptype["is_rate"] else data[ptype["col_y"]]
                    
                    ax.plot(data["Acceptance_Rate"], y_val, 
                            label=phase, color=colors[phase], marker="o", linewidth=2)
                    
                    if ptype["is_rate"]:
                         ci_l = data["KEP_MatchRate_CI_Lower"] * 100
                         ci_u = data["KEP_MatchRate_CI_Upper"] * 100
                         ax.fill_between(data["Acceptance_Rate"], ci_l, ci_u, color=colors[phase], alpha=0.1)

                ax.set_title(cat, fontsize=12, fontweight="bold")
                ax.set_xticks(ACCEPTANCE_RATES)
                ax.grid(True, linestyle="--", alpha=0.6)
                
                if idx == 0:
                    ax.set_ylabel(ptype["ylabel"])
                    ax.legend(fontsize="small")

            fig.text(0.5, 0.02, "Acceptance Rate (%)", ha="center", fontsize=12)
            plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
            
            filename = f"compare_early_late_demographics_{suffix}_{ptype['file_tag']}.png"
            plt.savefig(os.path.join(comp_output_dir, filename), dpi=300)
            plt.close()
            print(f"      -> Created Plot: {filename}")

    # Wait Times to compare: Global Average Wait Time
    df_wait_first = load_combined_data(policy_key, year, "waittime_general_summary.csv", sub_dir="first_20_epochs")
    df_wait_last = load_combined_data(policy_key, year, "waittime_general_summary.csv", sub_dir="last_20_epochs")

    if not df_wait_first.empty and not df_wait_last.empty:
        df_wait_first["Phase"] = "First 20 Epochs"
        df_wait_last["Phase"] = "Last 20 Epochs"
        df_wait_combined = pd.concat([df_wait_first, df_wait_last], ignore_index=True)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df_wait_combined, x="Acceptance_Rate", y="Mean_Wait_GlobalAvg", 
            hue="Phase", style="Phase", markers=True, dashes=False, linewidth=2.5,
            palette={"First 20 Epochs": "#009E73", "Last 20 Epochs": "#D55E00"}
        )
        
        plt.xlabel("Acceptance Rate (%)")
        plt.ylabel("Average Wait Time (Days)")
        plt.xticks(ACCEPTANCE_RATES)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Simulation Phase")
        
        filename = "compare_early_late_waittime.png"
        plt.savefig(os.path.join(comp_output_dir, filename), dpi=300)
        plt.close()
        print(f"      -> Created Plot: {filename}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    selection = None
    
    # 1. Argument Mode
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Plot KEP Results")
        parser.add_argument("policy", choices=list(POLICY_CONFIGS.keys()) + ["all", "compare"], help="Which policy to plot or compare")
        args = parser.parse_args()
        selection = args.policy

    # 2. Interactive Mode
    else:
        print("\n--- KEP Plotting Selector ---")
        options = list(POLICY_CONFIGS.keys()) + ["all"]

        for i, opt in enumerate(options, 1):
            name = "Plot All Policies (Individual)" if opt == "all" else POLICY_CONFIGS[opt]["name"]
            print(f"{i}. {name} ({opt})")
        
        compare_index = len(options) + 1
        print(f"{compare_index}. Compare All Policies")
        
        try:
            choice = int(input("\nSelect a number to plot: "))
            
            if choice == compare_index:
                selection = "compare"
            elif 1 <= choice <= len(options):
                selection = options[choice - 1]
            else:
                print("Invalid selection.")
                exit()
        except ValueError:
            print("Please enter a number.")
            exit()

    # 3. Execution Logic
    if selection == "compare":
        # Function for generating all comparison plots between policies
        generate_comparison_plots()
        
    elif selection == "all":
        # Generates all individual plots for each policy (without comparisons)
        for key in POLICY_CONFIGS.keys():
            generate_plots(key)
            
    elif selection:
        # Generates plots for a specific policy
        generate_plots(selection)