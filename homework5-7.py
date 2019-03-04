#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:38:04 2019

@author: jinyi
"""

import numpy as np

cur_a = np.array([1,1])
gamma = 0.1 # step size multiplier
itr = 0
precision = np.float_power(10, -14)
max_iters = 10000 # maximum number of iterations

def err(x):
    u, v = x
    result = (u*np.exp(v) - 2*v*np.exp(-u))**2
    return result
def dir_u(x):
    u, v = x
    result = 2*(np.exp(v) + 2*v*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))
    
    return result  
def dir_v(x):
    u, v = x
    result = 2*(u*np.exp(v) - 2*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))  
    return result


for i in range(15):
    
    prev_a = cur_a
    cur_a = prev_a -  gamma*np.array([dir_u(prev_a), 0])
    cur_a = prev_a -  gamma*np.array([0, dir_v(prev_a)])
#    print(err(prev_a), err(cur_a))

print(err(cur_a))    
