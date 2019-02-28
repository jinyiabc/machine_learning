#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:09:29 2019

@author: jinyi
"""

import numpy as np
# machine limits epsilon
eps = np.finfo(float).eps
N = 1000
sum = 0
sum1 = 0
for j in range(1):
    # create N= 1000 points
    data_set = np.random.uniform(low=-1, high=1.0+eps, size=(N,2))
    x1 = data_set[:,0]
    x2 = data_set[:,1]
    X_N = np.column_stack((x1 ,x2,(np.ones(N))))
    Y_N = np.sign(x1**2 + x2**2 - 0.6)
    
    def noise(y):
        # randomly select 10% of index of sets.
        noise_idx = np.random.choice(N, int(N*0.1))
    #    print("noise idx",noise_idx)
        y1 = np.zeros(N)
    #    y1[0] = -y[0]
        for j in range(N):
            y1[j] = y[j]
            for i in noise_idx:
                if j == i :          
                    y1[i] = y[i]*(-1)
    #    print(y,y1)
        return y1

    def noise1(y):
        y1 = np.zeros(N)
        noise_idx = np.random.choice([1, -1], N, replace=True, p=[0.9, 0.1])
        for idx, value in enumerate(noise_idx):
            y1[idx] = value*y[idx]
        return y1
#    test_for_noise1 = np.ones(N)
#    print(np.count_nonzero(noise1(test_for_noise1) == test_for_noise1))
    Y_N1 = noise1(Y_N)
#    test = np.count_nonzero(Y_N1 == Y_N)
#    print(test)
    
    # m1*x1 + m2*x2 + c1 = y
    #A1 = np.vstack([x1, x2, np.ones(N)]).T    A1 = X_N
    m1, m2, m0 = np.linalg.lstsq(X_N, Y_N1, rcond=None)[0]
    g_z = np.sign( m1*x1 + x2*m2 + c1) 
    E_in = (N - np.count_nonzero(g_z == Y_N1))/N
    sum = sum + E_in      
    
    X_N1 = np.column_stack(((np.ones(N)), x1 ,x2, x1*x2, x1**2, x2**2))
    c0, c1, c2, c3, c4, c5 = np.linalg.lstsq(X_N1, Y_N1, rcond=None)[0]
    g_f = np.sign( c0 + c1*x1 + c2*x2 + c3*x1*x2 + c4*x1**2 + c5*x2**2) 
    E_out = (N - np.count_nonzero(g_f == Y_N1))/N
#    print(c0, c1, c2, c3, c4, c5)
    sum1 = sum1 + E_out
print(sum/1000)
print(sum1/1000)

        
 
