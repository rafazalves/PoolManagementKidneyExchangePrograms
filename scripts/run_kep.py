import argparse
import os
import sys

from kep.constants import *

# Import simulators
from kep.simulation import simulator, simulator_static

# Import all policies
from kep.policies import (
    policy_kep_hier_with_wt,
    policy_kep_hier_no_wt,
    policy_kep_max_cardinality,
    policy_kep_static,
) 

# Configuration Dictionary for each solver
POLICY_CONFIGS = {
    "hier_with_wt": {
        "func": policy_kep_hier_with_wt.solver,
        "folder": "kep_hier_with_wt",
        "name": "Hierarchical with WT",
        "is_static": False
    },
    "hier_no_wt": {
        "func": policy_kep_hier_no_wt.solver,
        "folder": "kep_hier_no_wt",
        "name": "Hierarchical No WT",
        "is_static": False
    },
    "max_cardinality": {
        "func": policy_kep_max_cardinality.solver,
        "folder": "kep_max_cardinality",
        "name": "Max Cardinality",
        "is_static": False
    },
    "static": {
        "func": policy_kep_static.solver,
        "folder": "kep_static",
        "name": "Static",
        "is_static": True  # Flag to use simulator_static
    }
}

def run_simulation(solver_key):
    if solver_key not in POLICY_CONFIGS:
        print(f"Error: Solver '{solver_key}' not found. Choose from: {list(POLICY_CONFIGS.keys())}")
        return

    config = POLICY_CONFIGS[solver_key]

    # Simulation Settings
    n_instances = NUMBER_OF_INSTANCES
    simulyears = SIMULATION_YEARS
    max_cycle = MAX_CYCLE_LENGTH
    max_chain = MAX_CHAIN_LENGTH
    matching_period_days = MATCHING_PERIOD_DAYS
    percentage_immuno = ACCEPTANCE_IMMUNOSUPPRESSION_PERCENTAGE

    # Dynamic output directory based on the chosen solver
    output_dir = f"./results/{simulyears}_year_simulation/{config['folder']}/{percentage_immuno}_acceptance/"

    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Running {config['name']} ---")
    print(f"Output Directory: {output_dir}")

    for instance_id in range(1, n_instances + 1):
        # Input file paths
        charact_path = f"./data/{simulyears}_year_simulation/{percentage_immuno}_percentage/pool_{instance_id}.csv"
        graph_path = f"./data/{simulyears}_year_simulation/arcs_{instance_id}.txt"

        # Output file paths
        results_file = os.path.join(output_dir, f"results_{instance_id}.csv")
        history_file = os.path.join(output_dir, f"history_{instance_id}.csv")
        matched_file = os.path.join(output_dir, f"matched_{instance_id}.csv")
        error_file = os.path.join(output_dir, f"error_{instance_id}.txt")

        # Clear files
        open(results_file, "w").close()
        open(history_file, "w").close()
        open(matched_file, "w").close()

        # Write header of results file
        with open(results_file, "w") as f:
            f.write("Instance,Period,MaxCycle,MaxChain,Name,Epoch,NNodes,SolverVerts,Transplants,Incompat,Unsolved,Cycles,Chains,Immuno,Timeouts,Compatible_Left,AvgWait,IterSolverTime\n")

        print(f"Instance {instance_id}...", end=" ", flush=True)

        # Execute the correct simulator
        if config['is_static']:
            # Static simulator for static solvers
            simulator_static(
                simulyears, charact_path, graph_path,
                max_cycle, max_chain,
                config['func'], # The solver function
                results_file, matched_file, history_file, error_file,
                instance_id, config['name']
            )
        else:
            # Dynamic simulator
            simulator(
                simulyears, charact_path, graph_path, matching_period_days,
                max_cycle, max_chain,
                config['func'], # The solver function
                results_file, matched_file, history_file, error_file,
                instance_id, config['name']
            )
        print("Done.")

if __name__ == "__main__":

    # Check if arguments were passed (Automation Mode)
    # Options: "hier_with_wt", "hier_no_wt", "max_cardinality", "static"
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Run KEP Simulation")
        parser.add_argument("solver", choices=list(POLICY_CONFIGS.keys()) + ["all", "all_dynamic"], help="Which solver to run")
        args = parser.parse_args()

        selection = args.solver

    # If no arguments passed (Interactive Mode)
    else:
        print("\n--- KEP Simulation Selector ---")
        options = list(POLICY_CONFIGS.keys()) + ["all", "all_dynamic"]

        for i, opt in enumerate(options, 1):
            if opt == "all":
                name = "Run ALL Solvers (Static + Dynamic)"
            elif opt == "all_dynamic":
                name = "Run ALL Dynamic Solvers (No Static)"
            else:
                name = POLICY_CONFIGS[opt]['name']

            print(f"{i}. {name} ({opt})")

        try:
            choice = int(input("\nSelect a number to run: "))
            if 1 <= choice <= len(options):
                selection = options[choice - 1]
            else:
                print("Invalid selection.")
                exit()
        except ValueError:
            print("Please enter a number.")
            exit()

    # Execution Logic
    if selection == "all":
        for key in POLICY_CONFIGS.keys():
            run_simulation(key)
            
    elif selection == "all_dynamic":
        # Loop through configs and run only if is_static is False
        for key, config in POLICY_CONFIGS.items():
            if not config['is_static']:
                run_simulation(key)
                
    else:
        run_simulation(selection)
