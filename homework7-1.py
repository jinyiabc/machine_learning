#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:55:01 2019

@author: jinyi
"""

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


c = np.ones(35)
x1 = df1['x1']
x2 = df1['x2']
x1_power = df1['x1']**2
x2_power = df1['x2']**2
x1x2 = df1['x1']*df1['x2']
x1_x2 = abs(df1['x1'] - df1['x2'])
x1__x2 = abs(df1['x1'] + df1['x2'])
y = df1['y']

A = np.column_stack((c, x1, x2, x1_power, x2_power, x1x2, x1_x2, x1__x2))

c = np.ones(250)
x1 = df2['x1']
x2 = df2['x2']
x1_power = df2['x1']**2
x2_power = df2['x2']**2
x1x2 = df2['x1']*df2['x2']
x1_x2 = abs(df2['x1'] - df2['x2'])
x1__x2 = abs(df2['x1'] + df2['x2'])
y_out = df2['y']
A_out = np.column_stack((c, x1, x2, x1_power, x2_power, x1x2, x1_x2, x1__x2))


range = np.array([3, 4, 5, 6, 7])
for k in range:
    A0 = A[:25,:(k+1)]
    y0 = y[:25]
    cur_w = np.linalg.lstsq(A0, y0, rcond=None)[0]

    A1 = A[25:,:(k+1)]
    y1 = y[25:]
    E_VAL = np.count_nonzero(np.sign(np.dot(A1, cur_w)) != y1)/10.0
    print("E_VAL:", k, E_VAL)  

    A2 = A_out[:,:(k+1)]
    y2 = y_out
    E_OUT = np.count_nonzero(np.sign(np.dot(A2, cur_w)) != y2)/250.0
    print("E_OUT:", k, E_OUT)