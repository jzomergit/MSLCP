import networkx as nx
import pandas as pd
import MLCP_to_APP
import MLCP
import MSLCP
import numpy as np
import warnings


def init(jobs_basis, mtypes):
    jobs = MLCP_to_APP._add_jobslist_info(jobs_basis, ["typeA", "typeB"], mtypes, 7, 19)

    jobdict = dict()

    for index, job in jobs.iterrows():
        minuteRange = pd.date_range(start = job.release.floor('min'), end = job.deadline.ceil('min'), freq ="1min", closed="left")
        minuteRange = [str(x) for x in minuteRange]

        duration = int(np.floor(job.duration*60))

        entry = {'minuteRange': minuteRange, 'duration': duration}

        jobdict[index] = entry

    rangelist = [entry['minuteRange'] for job, entry in jobdict.items()]
    minutelist = list(set().union(*rangelist))
    minutelist.sort()

    return jobdict, minutelist, jobs


def _init_from_file(file):
    mtypesdict = {'typeA': {'o': 24, 'v': 0.5},
              'typeB': {'o': 48, 'v': 1}}
    mtypes = MLCP._format_mtypes(mtypesdict)

    jobs_basis = pd.read_csv(file)

    jobs_basis["s_time"] = [pd.Timestamp(el) for el in jobs_basis.s_time]
    jobs_basis["e_time"] = [pd.Timestamp(el) for el in jobs_basis.e_time]

    jobs = MLCP_to_APP._add_jobslist_info(jobs_basis, ["typeA", "typeB"], mtypes, 7, 19)
    jobs = jobs[jobs["duration"]>0].reset_index()

    jobdict = dict()
    for index, job in jobs.iterrows():
        minuteRange = pd.date_range(start = job.release.floor('min'), end = job.deadline.ceil('min'), freq ="1min", closed="left")
        minuteRange = [str(x) for x in minuteRange]

        duration = int(np.floor(job.duration*60))

        entry = {'minuteRange': minuteRange, 'duration': duration}

        jobdict[index] = entry

    rangelist = [entry['minuteRange'] for job, entry in jobdict.items()]
    minutelist = list(set().union(*rangelist))
    minutelist.sort()

    return jobdict, minutelist, jobs


def setFlowGraph(jobdict, minutelist):
    G = nx.DiGraph()

    G.add_node('s')
    for job in jobdict.keys():
        G.add_node(job)
    for minute in minutelist:
        G.add_node(minute)
    G.add_node('t')

    for job in jobdict.keys():
        G.add_edge('s', job, capacity=jobdict[job]['duration'])
    for job in jobdict.keys():
        for minute in jobdict[job]['minuteRange']:
            G.add_edge(job, minute, capacity=1)
    for minute in minutelist:
        G.add_edge(minute, 't', capacity=1)

    return G


def findMaxFlow(G, jobdict, computeRes = True):
    # investigate solution
    flow_value, flow_dict = nx.maximum_flow(G, 's', 't')

    requiredFlow = sum([G['s'][job]['capacity'] for job in jobdict.keys()])
    feasible = requiredFlow == flow_value

    if computeRes:
        resdict = dict()
        for job in jobdict.keys():
            resdict[job] = [key for key, value in flow_dict[job].items() if value == 1]

        maxlen = max([len(value) for key, value in resdict.items()])
        for key, value in resdict.items():
            resdict[key] = list(np.pad(value, (0, maxlen - len(value))))
        resdf = pd.DataFrame.from_dict(resdict)

        return flow_value, flow_dict, requiredFlow, feasible, resdf
    else:
        return flow_value, flow_dict, requiredFlow, feasible, None


def findResidualGraph(G):
    R = nx.algorithms.flow.preflow_push(G, 's', 't')
    attrs = {(u,v) : {'residual_flow': R[u][v]["capacity"] - R[u][v]["flow"]} for (u, v) in R.edges}
    nx.set_edge_attributes(G=R, values = attrs)

    return R


def findReachabilityGraph(R):
    H = nx.DiGraph()
    for node in R.nodes:
        H.add_node(node)
    for u, v in R.edges:
        if R[u][v]["residual_flow"] > 0:
            H.add_edge(u, v)

    Hx = H.copy()
    Hx.remove_node('s')

    return H, Hx


def getInfeasibleJobCombinations(Hx, H):
    infcombis = []
    for node in H['s']:  # for all nodes that are still reachable from the source, i.e. that have not yet been completely scheduled
        des = nx.algorithms.descendants(Hx, node)
        path = {d for d in des if isinstance(d, int)}
        path.add(node)
        infcombis.append(path)

    infcombis.sort(key=lambda x: len(x))
    infcombis_reduced = []
    for combi in infcombis:
        add = True
        for combi_small in infcombis_reduced:
            if combi.issuperset(combi_small) or combi == combi_small:
                add = False
                break
        if add:
            infcombis_reduced.append(combi)

    return infcombis, infcombis_reduced


def _check_validity(infcombis_reduced, jobs, N, M, timeoutlimit_APP):
    infeasiblevector = []
    for combi in infcombis_reduced:
        combi = list(combi)
        jobssub = jobs.loc[combi]
        t_app_shift, prob_app, dvs_app, inp_app, out_app, res = MSLCP.s3_run_app(jobssub, N, M, timeoutlimit_APP)
        infeasiblevector.append(res == "infeasible")
    valid = np.all(infeasiblevector)
    if(not valid):
        warnings.warn("Feasible combination added as a cut!")

    return valid



def _draw(G, jobdict):
    e = [(u, v) for (u, v, d) in G.edges(data=True)]

    nodes = G.nodes
    G_withoutst = nx.subgraph(G, [x for x in nodes if x not in ['s','t']])

    pos = nx.bipartite_layout(G_withoutst, nodes = jobdict.nodes())
    pos['s'] = [-3, 0]
    pos['t'] = [3, 0]

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=e,
                           width=3)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')