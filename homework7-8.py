qu# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:18:40 2019

@author: yijin
"""

import numpy as np
# machine limits epsilon
eps = np.finfo(float).eps 
N = 10
N1 = 100
sum_err = 0
for i in range(1):
  
    data_set = np.random.uniform(low=-1, high=1.0+eps, size=(2,2)) 
    # y = m*x + c
    A = np.column_stack((data_set[:,0], np.ones(2)))
    y = data_set[:,1]
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    w_0 = np.array([c, m, -1])
    
    data_set = np.random.uniform(low=-1, high=1+eps, size=(N,2))
    x_set = np.column_stack((np.ones(N),data_set[:,0],data_set[:,1]))
    y_set = np.sign(np.matmul(x_set,w_0))
    
    w = np.zeros(3)
    
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
        if np.all(test):
            sum = sum + i + 1
            break
        else:
            w = optimize(x_set,y_set,w)
            continue
        
        
    
    #N1 = 10000
    out_of_sample = np.random.uniform(low=-1, high=1.0+eps, size=(N1, 2))    
    XX = np.column_stack((np.ones(N1), out_of_sample))
    Y_f = np.sign(np.matmul(XX, w_0))
    Y_g = np.sign(np.matmul(XX, w))
    err = np.count_nonzero(Y_f != Y_g)/N1
    sum_err = sum_err + err


print("E_out:", sum_err/1000)

