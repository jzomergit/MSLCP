import pandas as pd
import pulp


def _define_APP(jobs, N, M):
    J = list(jobs.index)
    r = jobs.release_real
    d = jobs.deadline_real
    v = jobs.duration

    N = list(range(1, N+1))
    M = list(range(1, M+1))

    s = {n: {m: pulp.LpVariable(('s' + str(n).zfill(2) + '_' + str(m).zfill(2)), lowBound=0, upBound=None, cat=pulp.LpContinuous) for m in M} for n in N}
    x = {n: {m : {j: pulp.LpVariable(('x' + str(n).zfill(2) + '_' + str(m).zfill(2) + '_' + str(j).zfill(2)), lowBound=0, upBound=1, cat=pulp.LpInteger) for j in J} for m in M} for n in N}

    y = {n: pulp.LpVariable(('y' + str(n).zfill(2)), lowBound=0, upBound=1, cat=pulp.LpInteger) for n in N}

    prob = pulp.LpProblem("SCHEDULER", pulp.LpMinimize)
    prob += pulp.lpSum(y)



    for n in N:
        for m in M:
            prob += pulp.lpSum([x[n][m][j] * r[j] for j in J]) <= s[n][m], ("2_releasetime_n=" + str(n) + "_m=" + str(m))
            prob += s[n][m] <= pulp.lpSum([x[n][m][j] * (d[j] - v[j]) for j in J]), ("2_deadlinetime_n=" + str(n) + "_m=" + str(m))

    for n in N:
        for m in M[0:len(M)-1]:
            prob += s[n][m+1] >= s[n][m] + pulp.lpSum([x[n][m][j] * v[j] for j in J]), ("3_nextmomenttime_n=" + str(n) + "_m=" + str(m))

    for j in J:
        prob += pulp.lpSum([x[n][m][j] for n in N for m in M]) == 1, ("4_assignmentofjobs_j=" + str(j))

    for n in N:
        for m in M:
            prob += pulp.lpSum([x[n][m][j] for j in J]) <= 1, ("5_assignmentofmoments_n=" + str(n) + "_m=" + str(m))

    for n in N:
        prob += pulp.lpSum([x[n][m][j] for m in M for j in J]) <= len(M) * len(J) * y[n], ("6_teamIsActive_n=" + str(n))

    dvs = {'x': x, 's': s, 'y': y}
    inp = {'N': N, 'M': M, 'J': J}

    return prob, dvs, inp


def _generate_output_APP(jobs, dvs, inp):
    N = inp["N"]
    M = inp["M"]
    J = inp["J"]

    x = dvs['x']
    s = dvs['s']

    map_job_to_n = {j : n for n in N for m in M for j in J if x[n][m][j].varValue == 1}
    map_job_to_m = {j: m for n in N for m in M for j in J if x[n][m][j].varValue == 1}

    out_tbl = jobs.copy()
    out_tbl["n"] = pd.Series(map_job_to_n)
    out_tbl["m"] = pd.Series(map_job_to_m)
    out_tbl["start"] = [s[row.n][row.m].varValue for index, row in out_tbl.iterrows()]
    out_tbl["end"] = [row.start + row.duration for index, row in out_tbl.iterrows()]
    out_tbl["s_time"] = out_tbl.apply(lambda x: pd.Timestamp("2018-04-10") + pd.Timedelta(hours = x.start), axis=1)
    out_tbl["e_time"] = out_tbl.apply(lambda x: pd.Timestamp("2018-04-10") + pd.Timedelta(hours = x.end), axis=1)

    out_tbl = out_tbl.sort_values(by = ["n", "m"])

    return out_tbl


def _print_APP_outcome(prob):
    status = pulp.LpStatus[prob.status]

    print(pulp.LpStatus[prob.status])
    if (prob.status == 1):
        for var in prob.variables():
            print(var.name, " = ", var.varValue)
        print("objective value = ", prob.objective.value())


def _solve_APP(jobs, N, M):

    if len(jobs)>M:
        M = len(jobs)
    M = min(M, len(jobs))

    prob_app, dvs_app, inp_app = _define_APP(jobs, N, M)
    prob_app.solve() # use standard PuLP solver
    # prob_app.solve(solver = pulp.solvers.GUROBI_CMD(options = [("LogToConsole", 0)])) # use GUROBI solver
    if prob_app.status == 1:
        out_app = _generate_output_APP(jobs, dvs_app, inp_app)
        return prob_app, dvs_app, inp_app, out_app
    else:
        return prob_app, dvs_app, inp_app, None
