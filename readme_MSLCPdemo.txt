###############################
MSLCP demonstration version 1.0
README
###############################

This set of scripts allow users to solve the Maintenance Scheduling and Location Choice Problem (MSLCP) themselves.

Several files are provided.
- motbl.csv contains synthetic rolling stock circluation data for 7 days, for 30 rolling stock units.
Each line contains a Maintenance Opportunity (MOs), with associated rolling stock unit number (trainnr), start time s, end time e and location l. Within each trainnr the MOs are listed in chronological order.

- In MSLCPdemo_main.py, the main parameters are set, the data is read, the model is run and some output is generated. The demo should take 4 iterations to terminate.

- MLCP, MLCP_to_APP, MSLCP, APP and RAPP contains all required methods either directly or indirectly invoked by main_MSLCPdemo.py .
Specifically it defines the LP model using the PuLP package and solves it.
The files MLCP, MSLCP and APP also contain solver calls (i.e. prob.solve() or prob_app.solve()), which default to the standard PuLP solver but can be changed to another solver such as Gurobi, if available. Solving using Gurobi yields significant runtime performance improvements and is therefore highly recommended.

The following packages were used during the development process: func-timeout (version 4.3.5), networkx (version 2.4), numpy (version 1.19.3), pandas (version 1.1.4), pulp (version 2.3.1). The scripts were run with Python version 3.7.7 .
