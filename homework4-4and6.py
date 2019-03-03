# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

sum_a = 0
sum_d = 0

for i in range(1000):
    
    eps = np.finfo(float).eps
    data = np.random.uniform(low=-1, high=1.0 + eps, size=(1, 2))
    y = np.sin(data*np.pi)
    
    rand_points = np.transpose(np.vstack((data, y)))
    
    # y = mx + c
    x = np.array(rand_points[:,0])
#    A = np.column_stack((np.ones(2), X))
    y = np.array(rand_points[:,1])
    a2 = x[0]**2 + x[1]**2
    a1 = -2*(x[0]*y[0] + x[1]*y[1])
    a0 = y[0]**2 + y[1]**2
    p = np.poly1d([a2, a1, a0])
    p1 = np.polyder(p)
    solution1 = np.roots(p1)
#    solution = np.roots(p)
#    m = np.polyfit(X, Y, 1, full=True)[0]
#    sum_c = sum_c + c
    sum_a = sum_a + solution1
# print(sum_c/1000)
# print(sum_m/1000)

print(sum_a/1000)

for j in range(1000):
    
    eps = np.finfo(float).eps
    data = np.random.uniform(low=-1, high=1.0 + eps, size=(1, 2))
    y = np.sin(data*np.pi)
    
    rand_points = np.transpose(np.vstack((data, y)))
    x = np.array(rand_points[:,0])
    y = np.array(rand_points[:,1])
    a2 = x[0]**2 + x[1]**2
    a1 = -2*(x[0]*y[0] + x[1]*y[1])
    a0 = y[0]**2 + y[1]**2
    p = np.poly1d([a2, a1, a0])
    p1 = np.polyder(p)
    a = np.roots(p1)
    #dif = (sum_a - a)**2

    coeff = np.array(([sum_a/1000]))
    a1 = np.array(([a]))
    # y = c + mx
    qty = 10
    x0 = np.arange(-1, 1, 2.0/(qty*1.0), dtype=float)
    # x = np.column_stack((np.ones(qty), x0))
    y_g = x0*coeff
    y_g1 = x0*a1
    dif = (y_g - y_g1)**2
    var = np.mean(dif)
    #print(y_g1)
    sum_d = sum_d + var

print(sum_d/1000)





