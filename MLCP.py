import pandas as pd
import pulp


def run_MLCP(motbl, mtypes, T, eps, L_D_max):
    """"
    Runs the MLCP model by formatting the input data, defining the MILP model and solving it.
    Returns the problem object, the decision variables (dvs) and input parameters (inp), a slightly formatted motbl and the solver status after solving the MILP.
    The prob.solve() call solves the prob instance using the standard, free, non-sophisticated PuLP solver. This line can be altered to use another solver such as Gurobi.
    """

    mtypes = _format_mtypes(mtypes)
    motbl = _simplify_motbl(motbl, min(mtypes.v))
    motbl = _format_motbl(motbl)

    prob, dvs, inp = _define_MLCP(motbl, mtypes, T, eps=eps, L_D_max=L_D_max, problemfile = None)
    prob.solve() # use standard PuLP solver
    # prob.solve(solver=pulp.solvers.GUROBI_CMD(options=[("LogToConsole", 0)])) # use GUROBI solver, if installed

    status = pulp.LpStatus[prob.status]

    return prob, dvs, inp, motbl, status


def _simplify_motbl(motbl, minimum_molength_hrs):
    """
    Returns a new motbl without the mos with a duration shorter than minmium_molength_hrs
    """

    vmin = minimum_molength_hrs
    motbl = motbl[motbl.e - motbl.s > vmin]

    return motbl


def _format_motbl(motbl):
    """
    Returns an motbl with the iterator variables i (for the train number, starting from 1) and j (for the MO number of a specific train, starting from 1)
    """

    motbl_new = motbl.copy()
    def func(x):
        gr = x.copy()
        gr["mo_id"] = pd.Series(list(range(1, len(gr)+1)), index=gr.index)
        return gr
    motbl_new = motbl_new.groupby("trainnr", sort=False).apply(lambda x: func(x))

    motbl_new["train_id"] = pd.factorize(motbl_new.trainnr)[0]

    motbl_new["train_id"] = motbl_new["train_id"] + 1
    motbl_new["mo_id"] = motbl_new["mo_id"]

    motbl_new = motbl_new[
        ["train_id", "mo_id", "s", "e", "d", "l",
         "s_time", "e_time"]]
    motbl_new.columns = ["i", "j", "s", "e", "d", "l",
                         "s_time", "e_time"]

    motbl_new = motbl_new.set_index(["i", "j"])
    return(motbl_new)


def _format_mtypes(mtypes):

    mtypes = pd.DataFrame.from_dict(mtypes, orient='index')
    return(mtypes)


def _moIsDuringDay(s, e, startday_hr, startnight_hr):
    cond1 = (s % 24 >= startday_hr and s % 24 < startnight_hr)
    cond2 = (e % 24 >= startday_hr and e % 24 < startnight_hr)
    cond3 = ((e-s) <= (startnight_hr-startday_hr))

    if cond1 & cond2 & cond3:
        return 1 # daytime interval
    else:
        return 0 # nighttime interval


def _define_MLCP(motbl, mtypes, T, eps, L_D_max, b = None, problemfile = None):
    """
    Defines the MILP model for the MLCP, including the decision variables, objective function and constraints, and outputs to the path specifeid in 'problemfile'.
    """
    I = list(motbl.index.get_level_values(0).unique())
    J = {i : list(motbl.loc[i].index) for i in I}
    K = list(mtypes.index)
    L = list(motbl.l.unique())

    if b is None: # if initial conditions not explicitly satisfied, assume all trains are as-good-as-new
        b = {i : {k: 0 for k in K} for i in I}

    e = motbl.e
    s = motbl.s
    d = motbl.d
    l = motbl.l
    o = mtypes.o
    v = mtypes.v

    V = {i : {j : {k : [p for p in J[i] if e[i][j] < s[i][p] <= e[i][j] + o[k]] for k in K} for j in J[i]} for i in I} # V_{ijk} for J in J_i
    for i in V:
        V[i].update({0: {k: [p for p in J[i] if s.loc[(i,p)] <= o[k] + b[i][k]] for k in K}}) # V_{i0k}

    prob = pulp.LpProblem("MLCP", pulp.LpMinimize)

    x = {i : {j : {k: pulp.LpVariable(('x' + str(i).zfill(2) + '_' + str(j).zfill(3) + '_' + str(k)), lowBound=0, upBound=1, cat=pulp.LpInteger) for k in K} for j in J[i]} for i in I}
    y_D = pulp.LpVariable.dicts('y_D', L, lowBound=0, upBound=1, cat=pulp.LpInteger)
    y_N = {j : 1 for j in L}
    obj_interpretative = pulp.LpVariable('obj_interpretative')
    obj_technical = pulp.LpVariable('obj_technical')

    prob += pulp.lpSum([x[i][j][k] * (1-d[i][j]) for i in I for j in J[i] for k in K]) + eps * pulp.lpSum([x[i][j][k] for i in I for j in J[i] for k in K]), "objective MLCP"

    prob += obj_interpretative == pulp.lpSum([x[i][j][k] * (1-d[i][j]) for i in I for j in J[i] for k in K]), "0_objective_interpretative_part"
    prob += obj_technical == eps * pulp.lpSum([x[i][j][k] for i in I for j in J[i] for k in K]), "0_objective_technical_part"

    for i in I:
        for k in K:
            prob += 1 <= pulp.lpSum([x[i][p][k] for p in V[i][0][k]]), ("1_firstactivity_i=" + str(i).zfill(2) + "_k=" + str(k))

    for i in I:
        for j in J[i]:
            for k in K:
                if e[i][j] + o[k] <= T:
                    prob += x[i][j][k] <= pulp.lpSum([x[i][p][k] for p in V[i][j][k]]), ("2_interval_i=" + str(i).zfill(2) + "_j="+str(j).zfill(3)+"_k="+str(k))

    for i in I:
        for j in J[i]:
            for k in K:
                prob += x[i][j][k] <= y_D[l[i][j]] * d[i][j] + y_N[l[i][j]] * (1-d[i][j]), ("3_location_i=" + str(i).zfill(2) + "_j="+str(j).zfill(3)+"_k="+str(k))

    for i in I:
        for j in J[i]:
            prob += pulp.lpSum([x[i][j][k] * v[k] for k in K]) <= e[i][j] - s[i][j], ("4_duration_i=" + str(i).zfill(2) + "_j="+str(j).zfill(3))

    prob += pulp.lpSum([y_D[l] for l in L]) <= L_D_max, "5_max_daytime_locations"

    if problemfile is not None:
        prob.writeLP(problemfile)


    inp = {'I':I, 'J': J, 'K':K, 'L': L, 'V': V, 'y_N': y_N, 'b': b, 'L_D_max': L_D_max,
           'e': e, 's': s, 'd': d, 'l': l, 'o': o, 'v':v, 'eps': eps}
    dvs = {'x': x, 'y_D': y_D, 'obj_interpretative': obj_interpretative, 'obj_technical': obj_technical}

    return prob, dvs, inp


def _getOutputMOs(motbl, prob, inp, dvs):
    """
    For a solved 'prob', gives a table with the resulting schedule for each rolling stock unit.
    """
    K = inp["K"]
    x = dvs["x"]

    output_mos = motbl.copy()
    output_mos = output_mos.reset_index()
    output_mos["s_date_only"] = output_mos.apply(
        lambda x: str(x.s_time.day) + "-" + str(x.s_time.month) + "-" + str(x.s_time.year), axis=1)
    output_mos["e_date_only"] = output_mos.apply(
        lambda x: str(x.e_time.day) + "-" + str(x.e_time.month) + "-" + str(x.e_time.year), axis=1)
    output_mos["s_time_only"] = output_mos.apply(lambda x: str(x.s_time.hour) + ":" + str(x.s_time.minute).zfill(2),
                                                 axis=1)
    output_mos["e_time_only"] = output_mos.apply(lambda x: str(x.e_time.hour) + ":" + str(x.e_time.minute).zfill(2),
                                                 axis=1)

    if prob.status == 1:  # only if optimal model solution was found
        for k in K:
            output_mos[("mtype_" + str(k))] = output_mos.apply(lambda tbl: x[tbl.i][tbl.j][k].varValue, axis=1)

    return output_mos


def _getOutputLocs(prob, inp, dvs):
    """
    For a solved 'prob', gives the locations open during daytime and nighttime.
    """

    L = inp["L"]
    y_D = dvs["y_D"]
    y_N = inp["y_N"]

    if prob.status == 1:
        output_locs = pd.DataFrame.from_dict({'l': L})
        output_locs["day"] = output_locs.apply(lambda tbl: y_D[tbl.l].varValue, axis=1)
        output_locs["night"] = output_locs.apply(lambda tbl: y_N[tbl.l], axis=1)
        return output_locs
    else:
        return None