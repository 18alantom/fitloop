import time

def ftime(t1:float,t2:float)->str:
    # To return formatted time upto ms
    t = t2-t1
    gm = time.gmtime(t)
    try:
        ms = str(t).split('.')[1][:3]
    except:
        ms = ""
    if gm.tm_hour > 0:
        fstr = "%H h %M m %S s"
    elif gm.tm_min > 0:
        fstr = "%M m %S s"
    else:
        fstr = "%S s"
    return time.strftime(fstr, gm) + f" {ms} ms"

def ptime(t:float)->str:
    # To return formatted time upto us
    # for profiler use
    gm = time.gmtime(t)
    try:
        fs = str(t).split('.')[1]
        ms = fs[:3]
        us = fs[3:6]
    except:
        ms = ""
    if gm.tm_hour > 0:
        fstr = "%H h %M m %S s"
    elif gm.tm_min > 0:
        fstr = "%M m %S s"
    else:
        fstr = "%S s"
    return time.strftime(fstr, gm) + f" {ms} ms {us} us"