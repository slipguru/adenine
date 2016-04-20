#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import seaborn as sns

from datetime import datetime
from itertools import product

palette = sns.color_palette("Set1")

def next_color():
    palette.append(palette.pop(0))
    return palette[-1]

def reset_palette():
    global palette
    palette = sns.color_palette("Set1")

# ensure_list = lambda x: x if type(x) == list else [x]
def ensure_list(x):
    return x if type(x) == list else [x]

def modified_cartesian(*args):
    """Modified Cartesian product.

    This function takes two (ore more) lists and returns their Cartesian product,
    if one of the two list is empty this function returns the non-empty one.

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
    nonempty = [ensure_list(arg) if len(ensure_list(arg)) > 0 else [None] for arg in args]

    # Cartesian product
    return [list(c) for c in product(*nonempty)]

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
    """Transform seconds into a formatted time string.

    Parameters
    -----------
    seconds : int
        Seconds to be transformed.

    Returns
    -----------
    time : string
        A well formatted time string.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def get_time():
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

def ensure_symmetry(X):
    """Ensure matrix symmetry.

    Parameters
    -----------
    X : numpy.ndarray
        Input matrix of precomputed pairwise distances.

    Returns
    -----------
    new_X : numpy.ndarray
        Symmetric distance matrix. Values are averaged.
    """
    if not (X.T == X).all():
        return (X.T + X) / 2.
