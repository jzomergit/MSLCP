import pandas as pd
import numpy as np
import pulp
import warnings
import time
import MLCP
import MLCP_to_APP
import APP
import RAPP
import func_timeout
import random

def _run_MSLCP(motbl, mtypes, T, eps, L_D_max, startday_hr, startnight_hr, shiftlist, N, M, cutMethod, ncuts,
               cutMethod_fallback, ncuts_fallback, timeoutlimit_app, checkValidity, stopMethod, stopThreshold):
    if cutMethod == "mincut":
        if cutMethod_fallback == None or ncuts_fallback == None:
            warnings.warn("Fallback cut methods need to be specified.")
            return

    t_total = time.time() # start of the MSLCP procedure (including define mlcp)

    info_list = []
    cutRecord = []
    stop = False
    it = 0

    t_define_mlcp, prob, dvs, inp = s0_define_mlcp(motbl, mtypes, T, eps, L_D_max)

    t0 = time.time() # start of the loop (excluding define mlcp)
    while (not stop):
        print("\nIteration " + str(it))
        t_it = time.time()
        capUse = dict()
        t_app = 0
        t_gen_cuts = 0
        t_add_cuts = 0
        t_app_shift_dict = dict()
        t_gen_cuts_shift_dict = dict()
        t_add_cuts_shift_dict = dict()
        finished = []

        t_mlcp, output_mos, nconstraints_init = s1_run_mlcp(prob, inp, dvs, motbl)
        used = output_mos.apply(lambda x: x['mtype_typeA'] + x['mtype_typeB'] > 0, axis=1)
        output_mos = output_mos[used]
        output_mos = output_mos.reset_index(drop=True)

        if shiftlist is None:
            shiftlist_it_all = _get_unique_shifts(output_mos, startnight_hr)
            shiftlist_it = [s for s in shiftlist_it_all if s[1] == 1] # take only daytime maintenance shifts

        else:
            shiftlist_it = shiftlist

        for shift in shiftlist_it:
            shift_str = str(shift[0]) + "_" + str(shift[1]) + "_" + str(shift[2])
            print(shift_str)

            t_create_jobs_list, jobs, jobs_basis = s2_create_jobs_list(output_mos, shift[0], shift[1], shift[2], startday_hr,
                                                           startnight_hr, mtypes)
            t_app_shift, prob_app, dvs_app, inp_app, out_app, res = s3_run_app(jobs, N, M, timeoutlimit_app)

            if res == "outOfTime":
                print("Shift " + str(shift[0]) + "_" + str(shift[1]) + "_" + str(
                    shift[2]) + " not solved; solving took too long.")
                t_gen_cuts_shift = 0
                t_add_cuts_shift = 0
                finished.append(True)
            elif res == "infeasible":  # capacity violation
                t_gen_cuts_shift, new_cuts, cutRecord = s4_gen_cuts(jobs_basis, cutMethod, ncuts, cutMethod_fallback, ncuts_fallback, N, M, mtypes, startday_hr, startnight_hr, timeoutlimit_app, cutRecord, it, shift_str, checkValidity)
                t_add_cuts_shift = s5_add_cuts_to_mlcp(new_cuts, prob, dvs)
                finished.append(False)
            else:  # feasible; no capacity violation
                t_gen_cuts_shift = 0
                t_add_cuts_shift = 0
                finished.append(True)

            t_app = t_app + t_app_shift
            t_app_shift_dict[shift_str] = t_app_shift
            t_gen_cuts = t_gen_cuts + t_gen_cuts_shift
            t_add_cuts_shift_dict[shift_str] = t_add_cuts_shift
            t_add_cuts = t_add_cuts + t_add_cuts_shift
            t_gen_cuts_shift_dict[shift_str] = t_gen_cuts_shift
            capUse[shift_str] = res

        t_update, stop, it = s6_update(it, t0, stopMethod, stopThreshold, finished)
        t_it = time.time() - t_it
        itInfo = {'it': it-1, 't_it': t_it, 't_mlcp': t_mlcp, 't_create_jobs_list': t_create_jobs_list,
                  't_app': t_app, 't_app_shift_dict': t_app_shift_dict, 't_gen_cuts': t_gen_cuts,
                  't_gen_cuts_shift_dict': t_gen_cuts_shift_dict,
                  't_add_cuts': t_add_cuts, 't_add_cuts_shift_dict': t_add_cuts_shift_dict, 't_update': t_update,
                  'mlcp_before_cuts': prob.objective.value(), 'mlcp_constraints_before_cuts': nconstraints_init,
                  'capUse_before_cuts': capUse, 'mlcp_constraints_after_cuts': prob.numConstraints()}

        info_list.append(itInfo)
        info_df = pd.DataFrame(info_list)

    t_total = time.time() - t_total

    capUse_df = pd.DataFrame(list(info_df.capUse_before_cuts))
    cutRecord_df = pd.DataFrame(cutRecord)
    t_app_shift_df = pd.DataFrame(list(info_df.t_app_shift_dict))
    t_add_cuts_shift_df = pd.DataFrame(list(info_df.t_add_cuts_shift_dict))
    t_gen_cuts_shift_df = pd.DataFrame(list(info_df.t_gen_cuts_shift_dict))

    t_mlcp_final, output_mos_final, nconstraints_init_final = s1_run_mlcp(prob, inp, dvs, motbl)

    return info_df, capUse_df, cutRecord_df, t_app_shift_df, t_add_cuts_shift_df, t_gen_cuts_shift_df, output_mos_final, prob


def _generate_cuts(jobs_basis, cutMethod, ncuts, cutMethod_fallback, ncuts_fallback, N, M, mtypes, startday_hr, startnight_hr, timeoutlimit_app, cutRecord, it, shift_str, checkValidity):
    if cutMethod == "naive":
        cut, n_steps, njobs = _cutgen_naive(jobs_basis)
        cuts = [cut]
        cutRecord.append({'it': it, 'shift': shift_str, 'cutNo': 1, 'n_steps': n_steps, 'njobs': njobs, 'cutMethod': cutMethod})
    elif cutMethod == "heuristic_basic":
        cuts = []
        for itj in range(ncuts):
            print("cut " + str(itj))
            cut, n_steps, njobs = _cutgen_heuristic_basic(jobs_basis, N, M, mtypes, startday_hr, startnight_hr, timeoutlimit_app)
            cuts.append(cut)
            cutRecord.append({'it': it, 'shift': shift_str, 'cutNo': itj+1, 'n_steps': n_steps, 'njobs': njobs, 'cutMethod': cutMethod})
    elif cutMethod == "heuristic_binarysearch":
        cuts = []
        for itj in range(ncuts):
            print("cut " + str(itj))
            cut, n_steps, njobs = _cutgen_heuristic_binarysearch(jobs_basis, N, M, mtypes, startday_hr, startnight_hr, timeoutlimit_app)
            cuts.append(cut)
            cutRecord.append({'it': it, 'shift': shift_str, 'cutNo': itj + 1, 'n_steps': n_steps, 'njobs': njobs, 'cutMethod': cutMethod})
    elif cutMethod == "mincut":
        cuts, njobs_list, feasible, valid = _cutgen_mincut(jobs_basis, mtypes, checkValidity)
        if feasible or not valid:
            # resort to fallback cut method
            return _generate_cuts(jobs_basis, cutMethod_fallback, ncuts_fallback, None, None, N, M, mtypes, startday_hr, startnight_hr, timeoutlimit_app, cutRecord, it, shift_str, checkValidity)
        else:
            for itj in range(len(cuts)):
                cutRecord.append({'it': it, 'shift': shift_str, 'cutNo': itj+1, 'njobs': njobs_list[itj], 'cutMethod': cutMethod})
        print(str(len(cuts)) + ' cut(s) generated by mincut method')
    else:
        warnings.warn("Unknown cut generation method specified.")
        return None

    return cuts, cutRecord


def _cutgen_naive(jobs):
    cut = MLCP_to_APP._get_vars_from_jobslist(jobs, ["typeA", "typeB"])
    n_steps = 1
    njobs = len(cut)

    return cut, n_steps, njobs


def _cutgen_heuristic_basic(jobs_basis, N, M, mtypes, startday_hr, startnight_hr, timeoutlimit_app):
    jobs = MLCP_to_APP._add_jobslist_info(jobs_basis.copy(), ["typeA", "typeB"], mtypes, startday_hr, startnight_hr)
    new_jobs = pd.DataFrame(columns = jobs.columns)
    feasible = True
    n_steps = 0

    while not len(jobs) == 0 and feasible:
        n_steps = n_steps + 1

        moveIndexInt = random.randrange(len(jobs))
        moveIndexLoc = jobs.index[moveIndexInt]
        moveJob = jobs.loc[moveIndexLoc].copy()
        new_jobs = new_jobs.append(moveJob)
        jobs = jobs.drop(index=moveIndexLoc)

        t_app, prob_app, dvs_app, inp_app, out_app, res = s3_run_app(new_jobs, N, M, timeoutlimit_app)

        feasible = (res != "infeasible")
        print(feasible)

    cut = MLCP_to_APP._get_vars_from_jobslist(new_jobs, ["typeA", "typeB"])
    njobs = len(cut)

    return cut, n_steps, njobs


def _cutgen_heuristic_binarysearch(jobs_basis, N, M, mtypes, startday_hr, startnight_hr, timeoutlimit_app):
    jobs = MLCP_to_APP._add_jobslist_info(jobs_basis, ["typeA", "typeB"], mtypes, startday_hr, startnight_hr)
    A = pd.DataFrame(columns = jobs.columns)
    B = jobs.copy()

    n_steps = 0

    while len(B) > 1:
        n_steps = n_steps + 1
        B_left = pd.DataFrame(columns = jobs.columns)
        h = int(np.ceil(0.5 * len(B)))
        for i in range(h):
            moveIndexInt = random.randrange(len(B))
            moveIndexLoc = B.index[moveIndexInt]
            moveJob = B.loc[moveIndexLoc].copy()
            B_left = B_left.append(moveJob)
            B = B.drop(index=moveIndexLoc)

        B_right = B.copy()

        t_app, prob_app, dvs_app, inp_app, out_app, res = s3_run_app(A.append(B_left), N, M, timeoutlimit_app)

        if res == "infeasible":  # infeasible left-hand-side
            B = B_left.copy()
        else: # feasible or outOfTime left-hand-side # note: res may be 'feasible' OR 'outOfTime'. If outOfTime, I assume the set of jobs is feasible (although this may be too conservative).
            A = A.append(B_left)
            B = B_right.copy()

    cut = MLCP_to_APP._get_vars_from_jobslist(A.append(B), ["typeA", "typeB"])
    njobs = len(cut)

    return cut, n_steps, njobs


def _cutgen_mincut(jobs_basis, mtypes, checkValidity):
    jobdict, minutelist, jobs = RAPP.init(jobs_basis, mtypes)
    G = RAPP.setFlowGraph(jobdict, minutelist)
    flow_value, flow_dict, requiredFlow, feasible, resdf = RAPP.findMaxFlow(G, jobdict, computeRes = False)
    valid = True

    if not feasible: # RAPP infeasible implies APP infeasible
        R = RAPP.findResidualGraph(G)
        H, Hx = RAPP.findReachabilityGraph(R)
        infcombis, infcombis_reduced = RAPP.getInfeasibleJobCombinations(Hx, H)
        print(infcombis_reduced)
        if checkValidity:
            valid = RAPP._check_validity(infcombis_reduced, jobs)

    cuts = []
    njobs_list = []
    if not feasible and valid: # RAPP cannot be used and we resort to earlier techniques
        for combi in infcombis_reduced:
            new_jobs = jobs.loc[list(combi)]
            cuts.append(MLCP_to_APP._get_vars_from_jobslist(new_jobs, ["typeA", "typeB"]))
            njobs_list.append(len(combi))

    if checkValidity:
        print("Valid = " + str(valid))

    return cuts, njobs_list, feasible, valid


def _add_cut_to_MLCP(prob, dvs, cut):
    x = dvs["x"]
    varsomit = [x[el['i']][el['j']][el['k']] for el in cut]
    prob += pulp.lpSum([1 - el for el in varsomit]) >= 1

    return prob


def _stopCriterion(it, t0, stopMethod, stopThreshold, finished):
    if np.array(finished).all(): # if all shifts are satisfied
        return True

    if stopMethod == "iterations":
        if it >= stopThreshold - 1:  # if it == stopThreshold-1, then in total stopThreshold iterations have been performed (including the iteration with id 0)
            return True
        else:
            return False
    elif stopMethod == "time":
        if time.time() - t0 >= stopThreshold:
            return True
        else:
            return False
    else:
        warnings.warn("Unknown stopMethod specified.")
        return True


def _get_unique_shifts(output_mos, startnight_hr):
    output_mos = output_mos.copy()
    output_mos["refDate"] = output_mos.apply(lambda x: MLCP_to_APP._refDate(x.e_time, x.d, startnight_hr), axis=1)

    all_shifts = [(row.l, row.d, str(row.refDate)) for index, row in output_mos.iterrows()]
    unique_shifts = np.unique(all_shifts, axis=0)
    res = []
    for shift in unique_shifts:
        res.append((shift[0], int(shift[1]), str(shift[2])))

    return res


def s0_define_mlcp(motbl, mtypes, T, eps, L_D_max):
    # (0) Define MLCP for first time
    t_define_mlcp = time.time()
    prob, dvs, inp = MLCP._define_MLCP(motbl, mtypes, T, eps, L_D_max)
    t_define_mlcp = time.time() - t_define_mlcp

    return t_define_mlcp, prob, dvs, inp


def s1_run_mlcp(prob, inp, dvs, motbl):
    # (1) RUN MLCP
    t_mlcp = time.time()

    # solve
    prob.solve() # use standard PuLP solver
    # prob.solve(solver=pulp.solvers.GUROBI_CMD(options=[("LogToConsole", 0), ("MIPGap", 0.00000000001)])) # solve using Gurobi
    output_mos = MLCP._getOutputMOs(motbl, prob, inp, dvs)
    t_mlcp = time.time() - t_mlcp

    nconstraints = prob.numConstraints()

    return t_mlcp, output_mos, nconstraints


def s2_create_jobs_list(output_mos, shiftLoc, shiftDuringDay, shiftDate, startday_hr, startnight_hr, mtypes):
    # (2) CREATE JOBS LIST (= INPUT TO APP)
    t_create_jobs_list = time.time()
    jobs_basis = MLCP_to_APP._get_jobs_in_shift(output_mos, shiftLoc, shiftDuringDay, shiftDate, startnight_hr)
    jobs = MLCP_to_APP._add_jobslist_info(jobs_basis, ["typeA", "typeB"], mtypes, startday_hr, startnight_hr)
    t_create_jobs_list = time.time() - t_create_jobs_list

    return t_create_jobs_list, jobs, jobs_basis


def s3_run_app(jobs, N, M, timeoutlimit):
    # (3) RUN APP AND CHECK IF CAP IS VIOLATED
    t_app = time.time()

    try:
        prob_app, dvs_app, inp_app, out_app = func_timeout.func_timeout(timeoutlimit, APP._solve_APP, (jobs, N, M))
        obj_app = prob_app.objective.value()
        outOfTime = False
    except:
        print("APP not solved within " + str(timeoutlimit) + " seconds.")
        outOfTime = True

    if outOfTime:
        res = "outOfTime"
    elif prob_app.objective.value() is None or prob_app.status != 1:
        res = "infeasible"
    else:
        res = prob_app.objective.value()

    t_app = time.time() - t_app

    if not outOfTime:
        return t_app, prob_app, dvs_app, inp_app, out_app, res
    else:
        return t_app, None, None, None, None, res


def s4_gen_cuts(jobs_basis, cutMethod, ncuts, cutMethod_fallback, ncuts_fallback, N, M, mtypes, startday_hr, startnight_hr, timeoutlimit_app, cutRecord, it, shift_str, checkValidity):
    # (4) IF CAP VIOLATION, THEN GENERATE CUTS
    t_gen_cuts = time.time()
    new_cuts, cutRecord = _generate_cuts(jobs_basis, cutMethod, ncuts, cutMethod_fallback, ncuts_fallback, N, M, mtypes, startday_hr, startnight_hr, timeoutlimit_app, cutRecord, it, shift_str, checkValidity)
    t_gen_cuts = time.time() - t_gen_cuts

    return t_gen_cuts, new_cuts, cutRecord


def s5_add_cuts_to_mlcp(new_cuts, prob, dvs):
    # (5) ADD CUTS TO MLCP
    t_add_cuts = time.time()
    for cut in new_cuts:  # add cuts
        prob = _add_cut_to_MLCP(prob, dvs, cut)
    t_add_cuts = time.time() - t_add_cuts

    return t_add_cuts


def s6_update(it, t0, stopMethod, stopThreshold, finished):
    # (6) UPDATE

    t_update = time.time()
    stop = _stopCriterion(it, t0, stopMethod, stopThreshold, finished)
    it = it + 1
    t_update = time.time() - t_update

    return t_update, stop, it


def get_init_location_capacities(output_mos, startday_hr, startnight_hr, mtypes, N, M):
    unique_shifts = _get_unique_shifts(output_mos, startnight_hr)

    df_list = []
    unique_day_shifts = [shift for shift in unique_shifts if shift[1] == 1]
    for shift in unique_day_shifts:
        t_shift = time.time()
        print(shift)
        jobs = MLCP_to_APP._get_jobs_in_shift(output_mos, shift[0], shift[1], shift[2], startnight_hr)
        if(len(jobs) > 0):
            jobs = MLCP_to_APP._add_jobslist_info(jobs, ["typeA", "typeB"], mtypes, startday_hr, startnight_hr)
            prob_app, dvs_app, inp_app, out_app = APP._solve_APP(jobs, N, M)
            if (prob_app.objective.value() == None) or prob_app.status != 1:
                feasible = False
            else:
                feasible = True
        else:
            feasible = True

        print(prob_app.objective.value())
        t_shift = time.time() - t_shift
        df_list.append({'location': shift[0], 'duringDay': shift[1], 'refDate': shift[2], 'nJobs': len(jobs),
                        'feasible': feasible, 'OF': prob_app.objective.value(), 'computationTime': t_shift})

    location_capacity = pd.DataFrame(df_list)

    return location_capacity

