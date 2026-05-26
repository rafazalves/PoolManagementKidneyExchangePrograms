import random
import time

from kep.compatibility_graph import Digraph
from kep.constants import *
from kep.utils import log_pair_history, read_characterization, read_graph, read_times


def simulator(simulyears, characterization_path, graph_path, t, max_cycle, max_chain, solver, results_file, matched_file, history_file, error_file, instance_number, simulation_name):

    start_clock_time = time.time()
    start_cpu_time = time.perf_counter()

    # Read input files
    characterization = read_characterization(characterization_path)
    V, A = read_graph(graph_path)
    times, min_time, max_time = read_times(characterization)
    simul_max_time = simulyears * 360

    epoch = 0
    nnodes = 0 # Number of nodes in current epoch
    main_params = [instance_number, t, max_cycle, max_chain, simulation_name, epoch, nnodes]

    removed_nodes = set() # Set to track matched/removed nodes

    # Start and end time of the matching period
    start_time = 1
    end_time = start_time + t - 1

    state = {}
    state["RUN"] = 0

    print ("Starting main loop\n")

    with open(history_file, "w") as f_hist:
        f_hist.write("PairID,Arrival,Departure,WaitTime,Status,MatchType,MatchGroup,Epoch,BloodPatient,BloodDonor,PRA,Compatibility,Description\n")
        while end_time <= simul_max_time:

            iter_start = time.time()
            epoch += 1
            timeout_count = 0 # Count of timeouts in this epoch
            compatible_left = 0 # Count of compatible pairs left unmatched in this epoch

            if LOG:
                print (f"Starting iteration {epoch}")

            temp_start = time.time()

            # 1. Process timeouts (pairs that have departed before the beginning of this epoch)
            for node in times:
                if node not in removed_nodes:
                    departure = times[node][1]
                    if start_time <= departure < end_time:
                        # Retrieve pair data directly using the real ID
                        pair_data = characterization[node]

                        if(pair_data[COL_PAIR_COMPATIBILITY] == 3):
                            log_pair_history(node, departure, "COMPATIBLE_LEFT", times, characterization, f_hist, epoch)
                            compatible_left += 1
                        else:
                            log_pair_history(node, departure, "TIMEOUT", times, characterization, f_hist, epoch)
                            timeout_count += 1
                        removed_nodes.add(node)

            # 2. Get nodes and arcs for current epoch (arrival <= end_time <= departure)
            epoch_nodes = [node for node in times if times[node][0] <= end_time <= times[node][1] and node not in removed_nodes]
            epoch_arcs = [arc for arc in A if arc[0] in epoch_nodes and arc[1] in epoch_nodes]

            main_params[-2] = epoch
            main_params[-1] = len(epoch_nodes)

            # Map IDs
            encoder = {}
            decoder = []

            c = 0
            for el in epoch_nodes:
                encoder[el] = c
                decoder.append(el)
                c += 1

            # Create temporary characterization with encoded IDs for this epoch only
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
            data["tbm"] = t
            data["simulyears"] = simulyears

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
                print (f"Iteration graph has {itG.nv} nodes and {itG.ne} arcs")
                print (f"Structures done in {temp_end - temp_start:.2f} s")

            # Call solver
            solver_start = time.time()

            # res: statistics, cycles: list of lists with the matches
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

            # 3. Process matched pairs
            mpairs = [el for c in cycles for el in c]
            matched_encoded_indices = set(mpairs)

            # Save matched pairs to matched_file
            with open(matched_file, "a") as f:
                for item in cycles:
                    s1 = str(epoch)
                    s2 = ",".join([str(decoder[u]) for u in item])
                    out_str = s1+":"+s2+"\n"
                    f.write(out_str)

            # Process solver results statistics
            sum_wait_time = 0
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

                    # Calculate wait time
                    arrival = times[real_id][0]
                    wait = end_time - arrival
                    sum_wait_time += wait
                    transplant_count += 1

                    # Save the pair result in history file
                    log_pair_history(real_id, end_time, "MATCHED", times, characterization, f_hist, epoch, match_type, group_id)
                    removed_nodes.add(real_id)

            avg_wait_time = 0.0
            if transplant_count > 0:
                avg_wait_time = sum_wait_time / transplant_count

            # 4 & 5. Process Unmatched: Immuno-Transplants and Timeouts
            immuno_count = 0
            for node in epoch_nodes:
                if node not in removed_nodes:
                    # Retrieve pair data directly using the real ID
                    pair_data = characterization[node]

                    # Check for Immuno-suppression Transplant: Half-Compatible (2)
                    if pair_data[COL_PAIR_COMPATIBILITY] == 2:
                        # Check wait time threshold
                        accepts_immuno = False
                        
                        # Get the threshold (number of epochs willing to wait)
                        threshold_wait = pair_data[COL_PATIENT_ACCEPT_IMSUP]

                        # 0 means they accept immediately without waiting more epochs
                        if threshold_wait == 0:
                            accepts_immuno = True
                            if LOG:
                                print(f"Node {node} (Half-Compat) -> accepted Immunosuppression immediately.")

                        # -1 means they never accept
                        elif threshold_wait != -1:
                            # Calculate how many epochs they have effectively waited
                            # (current_epoch - 1) - arrival_epoch_index
                            waited_epochs = (epoch - 1) - int(pair_data[COL_PAIR_ARRIVAL] / t)
                            
                            # Ensure non-negative
                            waited_epochs = max(0, waited_epochs)
                            
                            if waited_epochs >= threshold_wait:
                                accepts_immuno = True
                                if LOG:
                                    print(f"Node {node} (Half-Compat) waited {waited_epochs} epochs (threshold {threshold_wait}) -> accepted Immunosuppression.")


                        if accepts_immuno:
                            log_pair_history(node, end_time, "IMMUNO", times, characterization, f_hist, epoch)
                            immuno_count += 1
                            removed_nodes.add(node)
                            if LOG:
                                print(f"Node {node} (Half-Compat) unmatched -> doing Immunosuppression transplant.")

                        elif times[node][1] == end_time:
                            log_pair_history(node, end_time, "TIMEOUT", times, characterization, f_hist, epoch)
                            timeout_count += 1
                            removed_nodes.add(node)

                    # Check for Timeout
                    elif times[node][1] == end_time:
                        if(pair_data[COL_PAIR_COMPATIBILITY] == 3):
                            log_pair_history(node, end_time, "COMPATIBLE_LEFT", times, characterization, f_hist, epoch)
                            compatible_left += 1
                        else:
                            log_pair_history(node, end_time, "TIMEOUT", times, characterization, f_hist, epoch)
                            timeout_count += 1
                        removed_nodes.add(node)

            # 6. Write Results
            start_time = end_time + 1
            end_time = start_time + t - 1

            out_res = []
            for i in main_params:
                out_res.append(i)
            for i in res:
                out_res.append(i)

            out_res.append(immuno_count)
            out_res.append(timeout_count)
            out_res.append(compatible_left)
            out_res.append(round(avg_wait_time, 2))
            out_res.append(solver_duration)

            out_res_str = ", ".join(str(i) for i in out_res) + "\n"
            with open(results_file, "a") as f:
                f.write(out_res_str)

            iter_end = time.time()
            iter_total = iter_end - iter_start

            if LOG:
                print (f"{len(mpairs)} KEP matched pairs")
                print (f"{immuno_count} Immuno-suppression transplants")
                print (f"{timeout_count} Timeouts")
                print (f"Iteration {epoch} done in {iter_total:.2f} s\n")

        # 7. Process Remaining Pairs (Unmatched at End of Simulation)
        remaining_count = 0

        for node in times:
            # If the pair was never matched/removed then it remains unmatched at end of simulation
            if node not in removed_nodes:
                # The Departure time is the end of the simulation
                log_pair_history(node, simul_max_time, "UNMATCHED_END", times, characterization, f_hist, epoch)
                remaining_count += 1

        if LOG:
            print(f"{remaining_count} pairs that remained in the pool at end of simulation")

        total_clock_time = time.time() - start_clock_time
        total_cpu_time = time.perf_counter() - start_cpu_time

        print (f"Done in {total_clock_time} seconds")
        print (f"CPU used: {total_cpu_time}\n")
