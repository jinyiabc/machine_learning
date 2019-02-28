#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 00:30:34 2019

@author: jinyi
"""

import numpy as np
N = 10
N1 = 1000
sum = 0
sum1 = 0
sum2 = 0
# machine limits epsilon
eps = np.finfo(float).eps

for j in range(1000):
    data_set = np.random.uniform(low=-1, high=1+eps, size=(N,2))
    # randomize two points within x = [-1,1] y = [-1,1]
    two_points = np.random.uniform(low=-1, high=1+eps, size=(2,2))
  
    x = two_points[:,0]
    y = two_points[:,1]
    
    # y = m*x + c
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    res = np.linalg.lstsq(A, y, rcond=None)[1]/N
    # y1 = m1*x1 + c1    
    x1 = data_set[:,0]
    y1 = data_set[:,1]
    # Calculate y_n according to g function.
    target_z = np.sign(y1 - m*x1 - c) 
    z_n = np.sign(y1 - m*x1 - c) 
    
    A1 = np.vstack([x1, y1, np.ones(len(x1))]).T
    m1, m2, c1 = np.linalg.lstsq(A1, z_n, rcond=None)[0]
    #res1 = np.linalg.lstsq(A1, z_n, rcond=None)[1]/N
    g_z = np.sign( m1*x1/m2 + y1 + c1/m2) 

    
    
    E_in = (N - np.count_nonzero(g_z == z_n))/N
    sum = sum + E_in
    # Create N1 population points. 
    data_set1 = np.random.uniform(low=-1, high=1+eps, size=(N1,2))
    x2 = data_set1[:,0]
    y2 = data_set1[:,1]
    target_z2 = np.sign(y2 - m*x2 - c)
    g_z2 = np.sign(m1*x2/m2 + y2 + c1/m2)
    E_out = (N1 - np.count_nonzero(g_z2 == target_z2))/N1
    sum1 = sum1 + E_out
    
    w = np.array([c1, m1, m2])
    x_set = np.column_stack((np.ones(N),data_set[:,0],data_set[:,1]))
    y_set = z_n
    def optimize(x,y,w):
        "test"
        y_1 = np.sign(np.matmul(x,w))        
#        print(y_1)
        y_2 = 2*(y_1 == y)-1            
        for idx, value in enumerate(y_2):
            if value == -1:
                w = w + x[idx]*y[idx]
                return w
    
    for i in range(500):
        test = (np.sign(np.matmul(x_set,w)) == y_set)
#1        print(i,test)
        if np.all(test):
#            print("success",i+1)
#            print(w)
            sum2 = sum2 + i + 1
            break
        else:
            w = optimize(x_set,y_set,w)
            continue
       
    
average = sum/1000
average1 = sum1/1000
average2 = sum2/1000
print("E_in",average)
print("E_out",average1)
print("Iteration", average2)





