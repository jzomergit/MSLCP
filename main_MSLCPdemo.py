import pandas as pd
import MLCP
import MSLCP

# Specify parameters
mtypes_dict = {"typeA": {'o': 24, 'v': 0.5}, "typeB": {'o': 48, 'v': 1}} # maintenance type definition
T = 7*24 # planning horizon, cannot exceed the validity of the input data (which is in this case 7 days = 7*24 hours)
eps = 0.001 # technical parameter, see MILP definition
L_D_max = 5 # number of maintenance locations opened during daytime at maximum
startday_hr = 7 # start of the daytime time window (hour of day)
startnight_hr = 19 # start of the nighttime time window (hour of day)

N = 1 # number of teams
M = 24 # number of 'moments' per team per shift

stopMethod = "iterations" # the stop criterion to restrict maximum computation time: "iterations" or "time"
stopThreshold = 5 # either the number of iterations, or the number of seconds (dependent on stopMethod)

timeoutlimit_app = 10 # the maximum running time of APP to prevent excessive APP computation times, in seconds
checkValidity = False # perform extra validity check; used for debugging


cutMethod = "mincut" #  Specify the cutmethod: "mincut", "heuristic_basic", "heuristic_binarysearch" or "naive"
ncuts = 999 # Specify the number of cuts for the heuristic methods (for the naive and mincut methods, this variable is redundant)
cutMethod_fallback = "naive" # Only relevant if cutMethod == "mincut". Specify fallback cutmethod in case the mincut cut method does not yield a useful result
ncuts_fallback = 99 # Only relevant if cutMethod == "mincut" (and if the fallback cutmethod is a heuristic method, else it is redundant)

shiftlist = None # Specify the list of shifts for which the capacity requirement must be checked. Shifts are defined by a tuple (location, daytime or nighttime indicator, date). Use shifts=None to check for all shifts.
# shiftlist = [("QFA", 1, "1970-01-02")] # Example of shiftlist that causes the MSLCP to only solve the capacity for the shift at location QFA during daytime on 02-01-1970.

# Read MO table input, containing all MOs.
# Should contain a list of all MOs, with for each MO the trainnr, the MO start time s in hours after midnight of the first day,
# the MO end time e in hours after midnight of the first day, and the location l.
motbl_in = pd.read_csv("motbl.csv", index_col=0)

# Add some extra variables: the variable d indicates for each MO whether it is a daytime MO or not,
# and the s_time and e_time variables are timestamps converting the variables s and e to interpretable dates, assuming here the first day in the analysis is January 1st, 1970.
motbl_in["d"] = motbl_in.apply(lambda x: MLCP._moIsDuringDay(s=x.s, e=x.e, startday_hr=startday_hr, startnight_hr=startnight_hr), axis=1)
motbl_in["s_time"] = [pd.Timestamp("1970-01-01") + pd.Timedelta(row.s, unit="hours") for index,row in motbl_in.iterrows()]
motbl_in["e_time"] = [pd.Timestamp("1970-01-01") + pd.Timedelta(row.e, unit="hours") for index,row in motbl_in.iterrows()]

# Format mtypes
mtypes = MLCP._format_mtypes(mtypes_dict)

# Format motbl
motbl = MLCP._simplify_motbl(motbl_in, min(mtypes.v))
motbl = MLCP._format_motbl(motbl)



# Run MLCP once to obtain an initial capacity requirements (only for demonstration purposes, not necessary for the remainder of the script)
prob, dvs, inp = MLCP._define_MLCP(motbl, mtypes, T, eps, L_D_max)
prob.solve()
# prob.solve(solver=pulp.solvers.GUROBI_CMD(options=[("LogToConsole", 0), ("MIPGap", 0.00000000001)]))
output_mos = MLCP._getOutputMOs(motbl, prob, inp, dvs)
location_capacities = MSLCP.get_init_location_capacities(output_mos, 7, 19, mtypes, 3, M)


# Run MSLCP. Returns:
# info_df with data about all iterations,
# capUse_df with the required capacity in each shift in the shiftlist per iteration,
# cutRecord_df with information about all generated cuts,
# t_app_shift_df, t_add_cuts_shift_df, t_gen_cuts_shift_df with information about the time consumption for the major parts of the algorithm,
# and output_mos_final and prob_final with the output_mos and prob objects obtained in the last iteration
info_df, capUse_df, cutRecord_df, t_app_shift_df, t_add_cuts_shift_df, t_gen_cuts_shift_df, output_mos_final, prob_final = MSLCP._run_MSLCP(motbl, mtypes, T, eps, L_D_max, startday_hr, startnight_hr,
               shiftlist, N, M, cutMethod, ncuts, cutMethod_fallback, ncuts_fallback,
               timeoutlimit_app, checkValidity, stopMethod, stopThreshold)

print("MSLCP model run successfully; final objective value = " + str(prob_final.objective.value()))


