from kep.constants import *


def create_cf_constraints(nv, cycles, chains):
    cf_constraints = {}
    nc = len(cycles)
    nch = len(chains)
    for j in range(nv):
        cf_constraints[j] = [indc for indc in range(nc) if j in cycles[indc]]
        for inds in range(nch):
            if j in chains[inds]:
                cf_constraints[j].append(inds + nc)
    return cf_constraints

def create_global_wtime_coefs(run, cycles, chains, wp_pairs, characterization):
    wtimes = sorted(list(set(wp_pairs)), reverse=True)

    # Dictionary: Key = Waiting Time, Value = List of coefficients (1 or 0) for all vars
    lexcoefs = {}

    for wt in wtimes:
        tmpall = []

        # Coefficients for Cycles
        for c in cycles:
            val = 0
            for i in c:
                # Count pair if it matches current waiting time and is not altruist
                if (wp_pairs[i] == wt) and (characterization[i][COL_PAIR_DESCRIPTION] != 'altruist'):
                    val += 1
            tmpall.append(val)

        # Coefficients for Chains
        for s in chains:
            val = 0
            for i in s:
                if (wp_pairs[i] == wt) and (characterization[i][COL_PAIR_DESCRIPTION] != 'altruist'):
                    val += 1
            tmpall.append(val)

        if sum(tmpall) > 0:
            lexcoefs[wt] = tmpall
        else:
            lexcoefs[wt] = None

    return lexcoefs

def create_global_compatibility_coefs(run, cycles, chains, comp_pairs, characterization):
    comps = sorted(list(set(comp_pairs)), reverse=True)

    # Dictionary: Key = Compatibility Score, Value = List of coefficients (1 or 0) for all vars
    lexcoefs = {}

    for comp in comps:
        tmpall = []

        # Coefficients for Cycles
        for c in cycles:
            val = 0
            for i in c:
                # Count pair if it matches current compatibility score
                if comp_pairs[i] == comp:
                    val += 1
            tmpall.append(val)

        # Coefficients for Chains
        for s in chains:
            val = 0
            for i in s:
                if comp_pairs[i] == comp:
                    val += 1
            tmpall.append(val)

        if sum(tmpall) > 0:
            lexcoefs[comp] = tmpall
        else:
            lexcoefs[comp] = None

    return lexcoefs
