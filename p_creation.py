#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:30:22 2019

@author: jinyi
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import transpose, matmul, dot
N = 7291
C = 0.01
Q = 2
const1 = np.array([4])
df1 = pd.read_csv('train.csv')
#df2 = pd.read_csv('test.csv')

x1 = df1['intensity']
x2 = df1['symmetry']
x_n = np.column_stack((x1, x2))
y_n = df1['digit']

for const in const1:
    y_n = 2*(abs(y_n - const) <= 0.001)-1
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i<=j:
                t = (1 + dot(x_n[i], x_n[j]))**Q
                P[i][j] = y_n[i]*y_n[j]*t
    
            else:                
                P[i][j] = P[j][i]
    path = 'p' + str(const) + '.npy'     
    np.save(path, P) 
