# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:27:16 2019

@author: yijin
"""

import numpy as np
# machine limits epsilon
eps = np.finfo(float).eps 
 
data_set = np.random.uniform(low=-1, high=1.0+eps, size=(2,2)) 
# y = m*x + c
A = np.column_stack((data_set[:,0], np.ones(2)))
y = data_set[:,1]
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
A0 = np.column_stack((data_set, np.ones(2)))
w = np.array([m, -1, c])
print(np.dot(A0, w))
shuffle = np.random.choice(100, 100, replace=False)
for i in shuffle:
    print(i)