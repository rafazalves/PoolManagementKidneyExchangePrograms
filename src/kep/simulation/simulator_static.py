import time

from kep.compatibility_graph import Digraph
from kep.constants import *
from kep.utils import log_pair_history, read_characterization, read_graph, read_times


def simulator_static(simulyears, characterization_path, graph_path, max_cycle, max_chain, solver, results_file, matched_file, history_file, error_file, instance_number, simulation_name):

    start_clock_time = time.time()
    start_cpu_time = time.perf_counter()

    # Read input files
    characterization = read_characterization(characterization_path)
    V, A = read_graph(graph_path)
    times, min_time, max_time = read_times(characterization)
    simul_max_time = simulyears * 360

    # Since this is a static simulation, we consider all nodes at once in a single epoch
    epoch = 1

    print("Starting Static Simulation (All-in-One Pool)\n")

    with open(history_file, "w") as f_hist:
        f_hist.write("PairID,Arrival,Departure,WaitTime,Status,MatchType,MatchGroup,Epoch,BloodPatient,BloodDonor,PRA,Compatibility,Description\n")

        iter_start = time.time()

        temp_start = time.time()

        # Get all nodes and arcs in the input files
        epoch_nodes = list(times.keys())
        epoch_nodes.sort()

        epoch_nodes_set = set(epoch_nodes)
        epoch_arcs = [arc for arc in A if arc[0] in epoch_nodes_set and arc[1] in epoch_nodes_set]

        nnodes = len(epoch_nodes)

        if LOG:
            print(f"Pool size: {nnodes} pairs")

        main_params = [instance_number, "STATIC", max_cycle, max_chain, simulation_name, epoch, nnodes]

        # Map IDs
        encoder = {}
        decoder = []

        c = 0
        for el in epoch_nodes:
            encoder[el] = c
            decoder.append(el)
            c += 1

        # Create temporary characterization with encoded IDs
        temp_charact = {}
        for el in epoch_nodes:
            temp = list(characterization[el])
            temp[COL_DONOR_ID] = encoder[temp[COL_DONOR_ID]]
            if temp[COL_PAIR_DESCRIPTION] != "altruist":
                temp[COL_PATIENT_ID] = encoder[temp[COL_PATIENT_ID]]
            temp_charact[encoder[el]] = temp

        data = {}
        data["characterization"] = temp_charact
        data["epoch"] = epoch

        # Build graph for solver
        itV = range(c)
        itA = [[] for i in range(c)]

        for arc in epoch_arcs:
            orig = arc[0]
            dest = arc[1]
            itA[encoder[orig]].append(encoder[dest])

        for el in itA:
            el.sort()

        itG = Digraph()
        itG.nv = len(itV)
        itG.ne = len(epoch_arcs)
        itG.verts = itV
        itG.adjList = itA

        temp_end = time.time()

        if LOG:
            print (f"Graph constructed: {itG.nv} nodes and {itG.ne} arcs")
            print (f"Structures done in {temp_end - temp_start:.2f} s")

        # Call solver
        solver_start = time.time()

        # State is irrelevant in static simulation
        state = {"RUN": 0}

        res, cycles = solver(itG, data, max_cycle, max_chain, state)

        solver_end = time.time()
        solver_duration = solver_end - solver_start

        # Handle solver errors
        if isinstance(res, int) and res == -1:
            print ("\n#### Error detected: Abort. ####\n")
            with open(error_file, "a") as f:
                f.write("%d %d\n" %(instance_number, len(cycles) if isinstance(cycles, list) else 0))
            return

        if LOG:
            print (f"Solver done in {solver_duration:.2f} s")

        mpairs = [el for c in cycles for el in c]
        matched_encoded_set = set(mpairs)
        removed_nodes = set()

        # Save matched pairs to matched_file
        with open(matched_file, "a") as f:
            for item in cycles:
                s1 = str(epoch)
                s2 = ",".join([str(decoder[u]) for u in item])
                out_str = s1+":"+s2+"\n"
                f.write(out_str)

        transplant_count = 0
        group_counter = 0

        for group in cycles:
            group_counter += 1
            # Identify Chain or Cycle (If the 1st node is Altruist = Chain)
            first_node_real_id = decoder[group[0]]
            is_chain = (characterization[first_node_real_id][COL_PAIR_DESCRIPTION] == "altruist")

            match_type = "Chain" if is_chain else "Cycle"
            group_id = f"{epoch}_{group_counter}"

            for encoded_idx in group:
                real_id = decoder[encoded_idx]
                transplant_count += 1

                log_pair_history(real_id, simul_max_time, "MATCHED", times, characterization, f_hist, epoch, match_type, group_id)
                removed_nodes.add(real_id)

        # Process Immuno-Transplants and Unmatched
        immuno_count = 0
        compatible_left_count = 0
        remaining_count = 0
        for node in epoch_nodes:
            if node not in removed_nodes:
                pair_data = characterization[node]

                if pair_data[COL_PAIR_COMPATIBILITY] == 2 and pair_data[COL_PATIENT_ACCEPT_IMSUP] > -1:
                    log_pair_history(node, simul_max_time, "IMMUNO", times, characterization, f_hist, epoch)
                    immuno_count += 1
                elif pair_data[COL_PAIR_COMPATIBILITY] == 3: # Fully compatible left without match
                    log_pair_history(node, simul_max_time, "COMPATIBLE_LEFT", times, characterization, f_hist, epoch)
                    compatible_left_count += 1
                else:
                    log_pair_history(node, simul_max_time, "UNMATCHED_END", times, characterization, f_hist, epoch)
                    remaining_count += 1
                removed_nodes.add(node)

        # Write results
        out_res = []
        for i in main_params:
            out_res.append(i)
        for i in res:
            out_res.append(i)

        out_res.append(immuno_count)
        out_res.append(0) # Timeout count (since static, always 0)
        out_res.append(compatible_left_count)
        out_res.append(0.0) # Avg wait time (since static, always 0.0)
        out_res.append(solver_duration)

        out_res_str = ", ".join(str(i) for i in out_res) + "\n"
        with open(results_file, "a") as f:
            f.write(out_res_str)

        iter_end = time.time()
        iter_total = iter_end - iter_start

        if LOG:
            print (f"Iteration done in {iter_total:.2f} s\n")
            print (f"{len(mpairs)} nodes matched in structures (KEP matches)")
            print (f"{immuno_count} pairs had immuno-transplants")
            print (f"{remaining_count} unmatched pairs")

        print (f"Total simulation time: {time.time() - start_clock_time:.2f} s")
        print (f"CPU used: { time.perf_counter() - start_cpu_time:.2f}\n")
