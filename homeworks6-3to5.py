#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:27:19 2019

@author: jinyi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:36:23 2019

@author: yijin
"""

import numpy as np
import pandas as pd
df1 = pd.read_csv('in.csv')
df2 = pd.read_csv('out.csv')

#df1['x1']
#df1.loc[32]
c = np.ones(35)
x1 = df1['x1']
x2 = df1['x2']
x1_power = df1['x1']**2
x2_power = df1['x2']**2
x1x2 = df1['x1']*df1['x2']
x1_x2 = abs(df1['x1'] - df1['x2'])
x1__x2 = abs(df1['x1'] + df1['x2'])
y = df1['y']
# df2 = pd.DataFrame(np.column_stack((c, x1, x2, x1_power, x2_power, x1x2, x1_x2, x1__x2)), columns=['1', 'x1', 'x2', 'x1_power', 'x2_power', 'x1x2', 'x1-x2', 'x1+x2'])

A = np.column_stack((c, x1, x2, x1_power, x2_power, x1x2, x1_x2, x1__x2))
# A*w = y
#E_IN = np.count_nonzero(np.sign(np.dot(A, w)) != y)/35.0
#print(E_IN)
#k = -1
    
def err(z, y, w, k):
    result = np.dot(np.transpose(np.dot(z, w) - y), (np.dot(z, w) - y))/(y.size)
    lamda = 10**k
    #lamda = 0
    result+=lamda/y.size*np.dot(w, w)
    return result
    
def der(z, y, w):
    result = np.dot(np.transpose(z), (np.dot(z, w) - y))/(y.size)
    return result 

k = 1
for k in range(10): 
    cur_w = np.linalg.lstsq(A, y, rcond=None)[0]
    pre_w = np.zeros(8)
    #cur_w = np.zeros(8)
    #print("initialization E_IN", err(A, y, cur_w, k))
    eta = 0.01
    precision = 10**(-7)
    N = y.size
    lamda = 10**(-k)
    max_iters = 10000
    itr = 0
    
    while np.linalg.norm((pre_w - cur_w)) >= precision and itr < max_iters:
        pre_w = cur_w
        cur_w = pre_w*(1-2*eta*lamda/N) - eta*der(A, y, pre_w)  
        itr+=1
    E_IN = np.count_nonzero(np.sign(np.dot(A, cur_w)) != y)/35.0
    print("E_IN:", E_IN)  
    #print("regression err:", err(A, y, cur_w, -k))  
    print("k:", -k, "iteration:", itr)
    
    c = np.ones(250)
    x1 = df2['x1']
    x2 = df2['x2']
    x1_power = df2['x1']**2
    x2_power = df2['x2']**2
    x1x2 = df2['x1']*df2['x2']
    x1_x2 = abs(df2['x1'] - df2['x2'])
    x1__x2 = abs(df2['x1'] + df2['x2'])
    y = df2['y']
    A = np.column_stack((c, x1, x2, x1_power, x2_power, x1x2, x1_x2, x1__x2))
    E_OUT = np.count_nonzero(np.sign(np.dot(A, cur_w)) != y)/250.0
    print("E_OUT:", E_OUT)
