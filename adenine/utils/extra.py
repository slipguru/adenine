#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import product

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
        
    
    
