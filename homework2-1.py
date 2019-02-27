#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 00:30:34 2019

@author: jinyi
"""

import numpy as np
N = 10
# machine limits epsilon
eps = np.finfo(float).eps
data_set = np.random.uniform(low=-1, high=1+eps, size=(N,2))
data_set = np.column_stack((np.ones(N), data_set[:,0], data_set[:,1]))
# randomize two points within x = [-1,1] y = [-1,1]
line = np.random.uniform(low=-1, high=1+eps, size=(2,2))
target_line = np.column_stack((np.ones(2),line[:,0],line[:,1]))

# define ax = b and solve the solution x, 
b = np.ones(2)
a = target_line[:,1:]
x = np.linalg.solve(a, b)
# target function
w_0 = np.concatenate(([-1],x))
# Calculate least square err.
