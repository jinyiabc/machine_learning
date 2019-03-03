#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:24:49 2019

@author: jinyi
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

sum_a = 0
sum_d = 0
sum_c = 0
sum_b = 0
sum_s = 0
sum_s1 = 0
sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0
sum5 = 0

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
    
    b = (y[0] + y[1])/2.0  
    # y = mx + c
    A = np.column_stack((np.ones(2), x))
    c_m = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # y = ax**2
    
    solution2 = (x[0]**4 + x[1]**4)/(2*((x[0]**2)*y[0] + (x[1]**2)*y[1]))
    # y = a*(x**2) + b
    A1 = np.column_stack((np.ones(2), x*x))
    b_a = np.linalg.lstsq(A1, y, rcond=None)[0]
#    solution = np.roots(p)
#    m = np.polyfit(X, Y, 1, full=True)[0]
#    sum_c = sum_c + c
    sum_a = sum_a + solution1
    sum_b = sum_b + b
    sum_c = sum_c + c_m   # y = mx +c
    sum_s = sum_s + solution2  # y = ax**2
    sum_s1 = sum_s1 + b_a # y = ax**2 + b



qty = 1000
x0 = np.arange(-1, 1, 2.0/(qty*1.0), dtype=float)
x01 = np.column_stack((np.ones(qty), x0))
x02 = np.column_stack((np.ones(qty), x0**2))

y_g1 = x0*(sum_a/1000)   # y = ax
y_g2 = x0*(sum_b/1000)   # y = b
y_g3 = np.matmul(x01, (sum_c/1000))  # y = c + mx
y_g4 = x0**2*(sum_s/1000)   # y = ax**2
y_g5 = np.matmul(x02, (sum_s1/1000))  # y = b + ax**2

y_f = np.sin(x0*np.pi)
bias1 = np.mean((y_g1 - y_f)**2) 
bias2 = np.mean((y_g2 - y_f)**2) 
bias3 = np.mean((y_g3 - y_f)**2) 
bias4 = np.mean((y_g4 - y_f)**2) 
bias5 = np.mean((y_g5 - y_f)**2) 

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

    a1 = np.array(([a]))
    
    y_g11 = x0*a1
    var1 = np.mean((y_g1 - y_g11)**2)
    
    y_g22 = (y[0] + y[1])/2.0
    var2 = np.mean((y_g2 - y_g22)**2)
    
    # y = mx + c
    A = np.column_stack((np.ones(2), x))
    c_m = np.linalg.lstsq(A, y, rcond=None)[0]
    y_g33 = np.matmul(x01, c_m)
    var3 = np.mean((y_g3 - y_g33)**2)
    
    # y = ax**2
    
    solution2 = (x[0]**4 + x[1]**4)/(2*((x[0]**2)*y[0] + (x[1]**2)*y[1]))
    y_g44 = x0**2*solution2
    var4 = np.mean((y_g4 - y_g44)**2)
    # y = a*(x**2) + b
    A1 = np.column_stack((np.ones(2), x*x))
    b_a = np.linalg.lstsq(A1, y, rcond=None)[0]
    y_g55 = np.matmul(x02, b_a)
    var5 = np.mean((y_g5 - y_g55)**2)

    
    sum1 = sum1 + var1
    sum2 = sum2 + var2
    sum3 = sum3 + var3
    sum4 = sum4 + var4
    sum5 = sum5 + var5


E_out1 = sum1/1000 + bias1
E_out2 = sum2/1000 + bias2
E_out3 = sum3/1000 + bias3
E_out4 = sum4/1000 + bias4
E_out5 = sum5/1000 + bias5

print("y=b", E_out2)
print("y = ax", E_out1)
print("y = c + mx", E_out3)
print("y = ax**2", E_out4)
print("y = b + ax**2", E_out5)


