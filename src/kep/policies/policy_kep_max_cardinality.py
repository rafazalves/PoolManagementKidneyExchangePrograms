import gurobipy as gp
from gurobipy import GRB

from kep.compatibility_graph import CycleFinder
from kep.constants import *
from kep.utils.utils_solver import *

# Global solver parameters
NUM_THREADS = 1
MAX_TIME = 900
MIP_GAP = 0.0

# Reverse graph flag (if digraph is in reverse order then 1 otherwise 0)
REVERSE = 0

def prepare_model(model):
    """
    To configure Gurobi model parameters.
    """
    model.setParam("LogToConsole", 0)
    model.setParam("TimeLimit", MAX_TIME)
    model.setParam("Threads", NUM_THREADS)
    model.setParam("MIPGap", MIP_GAP)

def add_cycle_chain_conflict_constraints(model, x, cf_constraints):
    """
    Constraints for pairs in cycle and chain can only be used once.
    """
    for node, var_indices in cf_constraints.items():
        if var_indices:
            model.addConstr(
                gp.quicksum(x[i] for i in var_indices) <= 1,
                name=f"cf_node_{node}"
            )

# Main Solver (called by simulator)
def solver(digraph, data, max_cycle, max_chain, state):

    if REVERSE == 1:
        digraph.reverse()

    if digraph.nv == 0:
        print("Empty graph, skipping optimization.")
        return [0, 0, 0, 0, 0, 0], []

    if LOG:
        print ("max_cycle:", max_cycle)
        print ("max_chain:", max_chain)

    characterization = data["characterization"]

    # Cycle and chain generation
    C = CycleFinder(digraph, characterization)

    if LOG:
        print ("Generating cycles...")
    C.find_cycles(max_cycle)

    if LOG:
        print ("Generating chains...")
    C.find_chains(max_chain)

    cycles = C.cycles
    chains = C.chains

    nc = len(cycles)
    nch = len(chains)

    # Pre-calculate constraints
    cf_constraints = create_cf_constraints(digraph.nv, cycles, chains)

    # Single Stage Optimization: Maximize number of transplants
    if LOG:
        print ("\nRunning Maximize Transplants Optimization...")

    model = gp.Model("Maximize_Transplants_Only")
    prepare_model(model)
    model.ModelSense = GRB.MAXIMIZE

    x = []

    # Cycle variables (only weight = number of transplants in the cycle)
    for i, cycle in enumerate(cycles):
        weight = len(cycle)
        x.append(model.addVar(vtype=GRB.BINARY, obj=weight, name=f"c_{i}"))

    # Chain variables (only weight = number of transplants in the chain - 1 because of altruist)
    for i, chain in enumerate(chains):
        weight = len(chain) - 1
        x.append(model.addVar(vtype=GRB.BINARY, obj=weight, name=f"s_{i}"))

    model.update()
    add_cycle_chain_conflict_constraints(model, x, cf_constraints)

    model.optimize()

    finalsol = [v.X for v in x]

    # Check solver status
    NumUnsolvedProbs = 0
    if model.status != GRB.OPTIMAL:
        NumUnsolvedProbs += 1
        print(f"Solver finished with status {model.status}")
        if model.status == GRB.INFEASIBLE:
            print("Model Infeasible!")

    selected_cycles_chains = []
    total_transplants = 0
    total_incompatible = 0

    num_cycles = 0
    num_chains = 0

    for i, val in enumerate(finalsol):
        if val > 0.5:
            if x[i].VarName.startswith("c_"):
                # IT IS A CYCLE
                idx = int(x[i].VarName.split("_")[1])
                obj = cycles[idx]

                selected_cycles_chains.append(obj)
                total_transplants += len(obj)
                num_cycles += 1

                if LOG:
                    print(f" > Selected Cycle ID {idx}: {obj}")

                for node in obj:
                    if characterization[node][COL_PAIR_COMPATIBILITY] == 1:
                        total_incompatible += 1

            elif x[i].VarName.startswith("s_"):
                # IT IS A CHAIN
                idx = int(x[i].VarName.split("_")[1])
                obj = chains[idx]

                selected_cycles_chains.append(obj)
                total_transplants += len(obj) - 1
                num_chains += 1

                if LOG:
                    print(f" > Selected Chain ID {idx}: {obj}")

                for node in obj[1:]:
                    if characterization[node][COL_PAIR_COMPATIBILITY] == 1:
                        total_incompatible += 1

    if LOG:
        print(f" > Optimization finished. Total Transplants: {total_transplants}")

    outputlist = [
        len(digraph.verts),
        total_transplants,
        total_incompatible,
        NumUnsolvedProbs,
        num_cycles,
        num_chains
    ]

    model.dispose()

    return outputlist, selected_cycles_chains
