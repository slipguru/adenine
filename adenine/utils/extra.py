#!/usr/bin/python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import os
import time
import seaborn as sns

from datetime import datetime
from itertools import product

palette = sns.color_palette("Set1")

def get_color(i=0):
    return palette[i]

def next_color():
    palette.append(palette.pop(0))
    return palette[-1]

def reset_palette(n_colors=6):
    global palette
    palette = sns.color_palette("Set1", n_colors)

# ensure_list = lambda x: x if type(x) == list else [x]
def ensure_list(x):
    return x if type(x) == list else [x]

def values_iterator(dictionary):
    '''Add support for python2 or 3 dictionary iterators. '''
    try:
        v = dictionary.itervalues() # python 2
    except:
        v = dictionary.values() # python 3
    return v

def items_iterator(dictionary):
    '''Add support for python2 or 3 dictionary iterators. '''
    try:
        gen = dictionary.iteritems() # python 2
    except:
        gen = dictionary.items() # python 3
    return gen

def modified_cartesian(*args, **kwargs):
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
    if kwargs.get('pipes_mode', False):
        nonempty = [ensure_list(arg) for arg in args if len(ensure_list(arg)) > 0]
    else:
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
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

def title_from_filename(root, step_sep="$\mapsto$"):
    # Define the plot title. List is smth like ['results', 'ade_debug_', 'Standardize', 'PCA']
    i = [i for i, s in enumerate(root.split(os.sep)) if 'ade_' in s][0]

    # lambda function below does: ('a_b_c') -> 'c b a'
    return step_sep.join(map(lambda x: ' '.join(x.split('_')[::-1]), root.split(os.sep)[i+1:]))

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

def timed(function):
    """Decorator that measures wall time of the decored function."""
    def timed_function(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        print("\nAdenine {} - Elapsed time : {} s\n".format(function.__name__, sec_to_time(time.time() - t0)))
        return result
    return timed_function

def set_module_defaults(module, dictionary):
    """Set default variables of a module, given a dictionary.
    Used after the loading of the configuration file to set some defaults."""
    for k, v in items_iterator(dictionary):
        try:
            getattr(module, k)
        except AttributeError:
            setattr(module, k, v)
