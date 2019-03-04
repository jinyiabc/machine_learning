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
    u = x[0]
    v = x[1]
    result = (u*np.exp(v) - 2*v*np.exp(-u))**2
    return result
def dir_u(x):
    u = x[0]
    v = x[1]
    result = 2*(np.exp(v) + 2*v*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))
    return result  
def dir_v(x):
    u = x[0]
    v = x[1]
    result = 2*(u*np.exp(v) - 2*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))  
    return result
while err(cur_a) >= precision and itr < max_iters:
    prev_a = cur_a
    cur_a = prev_a -  gamma*np.array([dir_u(prev_a), dir_v(prev_a)])
    # print(cur_a)
    itr+=1
    
print(itr, err(cur_a))
print(cur_a)