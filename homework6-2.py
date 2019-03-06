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
# w = np.zeros(8)
# A*w = y
w = np.linalg.lstsq(A, y, rcond=None)[0]
E_IN = np.count_nonzero(np.sign(np.dot(A, w)) != y)/35.0
print(E_IN)

def der(z, y, w):
    result = np.transpose(z)*(np.dot(z, w) - y)/(y.size)
    return result    

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
E_OUT = np.count_nonzero(np.sign(np.dot(A, w)) != y)/250.0
print(E_OUT)
