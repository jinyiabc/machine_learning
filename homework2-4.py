#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:34:16 2019

@author: jinyi
"""

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
sum1 = 0
for j in range(1000):
    # create N= 1000 points
    data_set = np.random.uniform(low=-1, high=1.0+eps, size=(N,2))
    x1 = data_set[:,0]
    x2 = data_set[:,1]
    X_N = np.column_stack((x1 ,x2,(np.ones(N))))
    Y_N = np.sign(x1**2 + x2**2 - 0.6)
    Y_N1 = noise1(Y_N)
    
    coeff = np.array([-0.98080674, 0.0241452, 0.00570705, 0.02873131, 1.57044171, 1.56941797])
    x = np.column_stack((np.ones(N) ,x1 , x2, x1*x2, x1**2, x2**2))
    g_f = np.sign(np.matmul(x, coeff)) 
    
    E_out = (N - np.count_nonzero(g_f == Y_N1))/N
    sum1 = sum1 + E_out


    
print("E_out:",sum1/1000)

def noise(y):
    # randomly select 10% of index of sets.
    noise_idx = np.random.choice(N, int(N*0.1))
#    print("noise idx",noise_idx)
    y1 = np.zeros(N)
#    y1[0] = -y[0]
    for j in range(N):
        y1[j] = y[j]
        for i in noise_idx :
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
 
