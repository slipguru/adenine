#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import product
import time

def modified_cartesian(*args):
    """Modified Cartesian product.

    This function takes two (ore more) lists and returns their Cartesian product, if one of the two list is empty this function returns the non-empty one.

    Parameters
    -----------
    *args : lists, length : two or more
        The group of input lists.

    Returns
    -----------
    cp : list
        The Cartesian Product of the two (or more) nonempty input lists.
    """
    # Get the non-empty input lists
    nonempty = []
    for arg in args:
        if len(arg)>0:
            nonempty.append(arg)
    # Cartesian product
    cp = []
    for c in product(*nonempty):
        cp.append(list(c))
    return cp

def make_time_flag():
    """Generate a time flag.

    This function simply generates a time flag using the current time.

    Returns
    -----------
    timeFlag : string
        A unique time flag.
    """
    y = str(time.localtime().tm_year)
    mo = str(time.localtime().tm_mon)
    d = str(time.localtime().tm_mday)
    h = str(time.localtime().tm_hour)
    mi = str(time.localtime().tm_min)
    s = str(time.localtime().tm_sec)
    return h+':'+mi+':'+s+'_'+d+'-'+mo+'-'+y


def sec_to_time(seconds):
    """Transform seconds into formatted time string"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)
