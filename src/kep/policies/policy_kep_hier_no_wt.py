import random

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

def optimize_lexicographic_global(cycles, chains, cf_constraints, lexcoefs):
    """
    Lexicographically maximize global compatibility.
    Higher compatibility have higher priority.
    """

    nc = len(cycles)
    nch = len(chains)
    nvars = nc + nch

    lex_levels = sorted(lexcoefs.keys(), reverse=True)

    model = gp.Model("Lexicographic_Global_Compatibility")
    prepare_model(model)

    # Decision variables
    x = model.addVars(nvars, vtype=GRB.BINARY, name="x")

    # Add constraints: each node used at most once
    add_cycle_chain_conflict_constraints(model, x, cf_constraints)

    model.ModelSense = GRB.MAXIMIZE

    lex_values = {}
    num_unsolved = 0

    # Lexicographic loop
    for lvl in lex_levels:

        coef = lexcoefs[lvl]
        if coef is None:
            continue

        model.setObjective(
            gp.quicksum(coef[i] * x[i] for i in range(nvars)),
            GRB.MAXIMIZE
        )

        model.optimize()

        if model.status != GRB.OPTIMAL:
            num_unsolved += 1
            if model.status == GRB.INFEASIBLE:
                raise RuntimeError(f"Infeasible at lex level {lvl}")
            continue

        # Save the result of this level to constrain the next levels
        opt_val = model.objVal
        lex_values[lvl] = opt_val

        model.addConstr(
            gp.quicksum(coef[i] * x[i] for i in range(nvars)) == opt_val,
            name=f"fix_lex_{lvl}"
        )

    sol = [x[i].X for i in range(nvars)]
    model.dispose()

    return sol, lex_values, num_unsolved

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

    # Calculate Compatibility scores for each pair
    compatibility_pairs = [0 for _ in digraph.verts]

    for i in digraph.verts:
        # base_score = 0 because we are only considering compatibility here (no waiting time)
        base_score = 0

        # 2. Add compatibility hierarchy bonus (Compatibility 1 = Incompatible, 2 = Half, 3 = Compatible)
        if characterization[i][COL_PAIR_COMPATIBILITY] == 1:
            base_score += (2 * HIERARCHY_MULTIPLIER)
        elif characterization[i][COL_PAIR_COMPATIBILITY] == 2:
            base_score += (1 * HIERARCHY_MULTIPLIER)

        compatibility_pairs[i] = base_score


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

    # Pre-calculate constraints and lex coefficients
    cf_constraints = create_cf_constraints(digraph.nv, cycles, chains)
    lexcoefs = create_global_compatibility_coefs(data["epoch"], cycles, chains, compatibility_pairs, characterization)

    # 1. Solve Lexicographic compatibility optimization
    if LOG:
        print ("\nRunning Global Lexicographic Compatibility Optimization...")

    lex_sol, optimal_compatibility_profile, NumUnsolvedProbs = optimize_lexicographic_global(cycles, chains, cf_constraints, lexcoefs)

    # 2. Solve Tie-Breaker (Maximize Incompatibles)
    if LOG:
        print ("\nRunning Tie-Breaker Optimization...")

    model = gp.Model("Final_TieBreaker")
    prepare_model(model)
    model.ModelSense = GRB.MAXIMIZE

    x = []

    # Initialize Random Seed for this run
    random.seed()

    # Cycle variables
    for i, cycle in enumerate(cycles):
        weight = len(cycle)

        for node in cycle:
            # Incompatible Pairs (Add Bonus)
            if characterization[node][COL_PAIR_COMPATIBILITY] == 1:
                weight += INCOMPATIBLE_BONUS
            # Half-Compatible Pairs (Add Bonus)
            elif characterization[node][COL_PAIR_COMPATIBILITY] == 2:
                weight += HALFCOMPATIBLE_BONUS

        # Add small random noise to weight to ensure different solutions in case of ties
        # This breaks ties between identical cycles without overriding the Bonuses (0.0001)
        weight += random.uniform(0, 0.0001)

        if LOG:
            print(f"[DEBUG] Cycle {i}: Final Weight={weight:.6f}")

        x.append(model.addVar(vtype=GRB.BINARY, obj=weight, name=f"c_{i}"))

    # Chain variables
    for i, chain in enumerate(chains):
        weight = len(chain) - 1

        for node in chain[1:]:
            # Incompatible Pairs (Add Bonus)
            if characterization[node][COL_PAIR_COMPATIBILITY] == 1:
                weight += INCOMPATIBLE_BONUS
            # Half-Compatible Pairs (Add Bonus)
            elif characterization[node][COL_PAIR_COMPATIBILITY] == 2:
                weight += HALFCOMPATIBLE_BONUS

        # Add small random noise to weight to ensure different solutions in case of ties (same as cycles)
        weight += random.uniform(0, 0.0001)

        if LOG:
            print(f"[DEBUG] Chain {i}: Final Weight={weight:.6f}")

        x.append(model.addVar(vtype=GRB.BINARY, obj=weight, name=f"s_{i}"))

    model.update()
    add_cycle_chain_conflict_constraints(model, x, cf_constraints)

    # Enforce lexicographic optimal profile
    for lvl, val in optimal_compatibility_profile.items():
        coef = lexcoefs[lvl]
        model.addConstr(
            gp.quicksum(coef[i] * x[i] for i in range(len(x))) == val,
            name=f"lex_fix_{lvl}"
        )

    model.optimize()

    finalsol = [v.X for v in x]

    # Extract Results
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
