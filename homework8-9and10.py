#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:30:58 2019

@author: jinyi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:35:45 2019

@author: jinyi
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import transpose, matmul, dot
from svmutil import *
#import random
#from datetime import datetime
#random.seed(datetime.now())

recog = 1.0


df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

array = [1.0, 5.0]
df1 = df1.loc[df1['digit'].isin(array)]


x1 = df1['intensity']
x2 = df1['symmetry']
x_n = np.column_stack((x1, x2))
y_n = df1['digit']


# CONST versus all classification.s
y_n = np.array(2*(abs(y_n - recog) <= 0.001)-1)

prob  = svm_problem(y_n, x_n)
range_c = np.array([0.01, 1, 100, 10000, 1000000])

# E_IN
for idx, value in enumerate(range_c):
    para = '-s 0 -c ' + str(value) + ' -t 2 -g 1 -r 2 -d 2 -q'
    param = svm_parameter(para)
    m = svm_train(prob, param)
    p_labels, p_acc, p_vals = svm_predict(y_n, x_n, m)
    
# E_OUT
#df2 = df2.loc[df2['digit'].isin(array)]
#
#x1 = df2['intensity']
#x2 = df2['symmetry']
#x_nn = np.column_stack((x1, x2))
#y_nn = df2['digit']
    
#y_nn = np.array(2*(abs(y_nn - recog) <= 0.001)-1)
#for idx, value in enumerate(range_c):
#    para = '-s 0 -c ' + str(value) + ' -t 2 -g 1 -r 2 -d 2 -q'
#    param = svm_parameter(para)
#    m = svm_train(prob, param)
#    p_labels, p_acc, p_vals = svm_predict(y_nn, x_nn, m)



