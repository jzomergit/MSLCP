import pandas as pd

def _refDate(e_time, daytime, startnight_hr):
    date = e_time.date()
    time = e_time.time()

    if daytime == 1:
        return date
    else:
        if time.hour >= startnight_hr:
            return date
        else:
            return date - pd.Timedelta(days=1)


def _release(s_time, e_time, daytime, refDate, durationhrs, startnight_hr):
    if daytime == 1: #daytime
        return s_time
    else: # nighttime
        startTimeWindow = pd.Timestamp(refDate) + pd.Timedelta(hours=startnight_hr)
        adjust1 = max(startTimeWindow, s_time)
        adjust2 = min(adjust1, e_time - pd.Timedelta(hours = durationhrs))
        return adjust2


def _deadline(s_time, e_time, daytime, refDate, durationhrs, startday_hr):
    if daytime == 1: #daytime
        return e_time
    else: # nighttime
        endTimeWindow = pd.Timestamp(refDate) + pd.Timedelta(days=1) + pd.Timedelta(hours=startday_hr)
        adjust1 = min(endTimeWindow, e_time)
        adjust2 = max(adjust1, s_time + pd.Timedelta(hours = durationhrs))
        return adjust2


def _get_jobs_in_shift(output_mos, location, duringDay, date, startnight_hr):
    # Given an MLCP solution (output_mos), get a list of all jobs in a given shift

    tbl = output_mos.copy()
    tbl["refDate"] = tbl.apply(lambda x: _refDate(x.e_time, x.d, startnight_hr), axis=1)

    if location != None:
        tbl = tbl[tbl["l"] == location]

    if date != None:
        tbl = tbl[tbl["refDate"] == pd.Timestamp(date)]

    if duringDay != None:
        tbl = tbl[tbl["d"] == duringDay]

    return tbl


def _relax_jobslist(jobs, keep):
    if keep == None:
        return jobs

    df = jobs.copy()
    df = df.set_index(["i", "j"], drop = True)

    df["mtype_typeA"] = 0
    df["mtype_typeB"] = 0
    for el in keep:
        df.at[(el['i'], el['j']), "mtype_" + el['k']] = 1

    df = df.reset_index()

    return df


def _add_jobslist_info(jobs_basis, K, mtypes, startday_hr, startnight_hr):
    # calculate deadlines etc
    jobs = jobs_basis.copy()
    jobs["duration"] = jobs.apply(lambda x: sum([x[("mtype_" + k)] * mtypes.loc[k]["v"] for k in K]), axis=1)
    jobs["isUsed"] = jobs.apply(lambda x: 1 if x["duration"] > 0 else 0, axis=1)
    jobs["release"] = jobs.apply(lambda x: _release(x.s_time, x.e_time, x.d, x.refDate, x.duration, startnight_hr), axis=1)
    jobs["deadline"] = jobs.apply(lambda x: _deadline(x.s_time, x.e_time, x.d, x.refDate, x.duration, startday_hr), axis=1)
    jobs["release_real"] = jobs.apply(lambda x: x.s + (x.release - x.s_time).total_seconds() / 60 / 60, axis=1)
    jobs["deadline_real"] = jobs.apply(lambda x: x.e + (x.deadline - x.e_time).total_seconds() / 60 / 60, axis=1)

    jobs = jobs.rename(columns = {'s': 's_original', 'e': 'e_original', 's_time': 's_time_original', 'e_time': 'e_time_original'})
    jobs = jobs.drop(columns = ['s_date_only', 's_time_only', 'e_date_only', 'e_time_only'])

    return jobs


def _get_vars_from_jobslist(jobs, K):
    lst = list()
    for index, row in jobs.iterrows():
        for k in K:
            if row[("mtype_"+ str(k))] == 1:
                lst.append({'i': row.i, 'j': row.j, 'k': k})

    return lst

