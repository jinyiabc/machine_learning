#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:12:05 2019

@author: jinyi
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import transpose, matmul, dot
from svmutil import *

#N = 7291
#C = 0.01
#Q = 2
recog = 0


df1 = pd.read_csv('train.csv')
#df2 = pd.read_csv('test.csv')

x1 = df1['intensity']
x2 = df1['symmetry']
x_n = np.column_stack((x1, x2))
y_n = df1['digit']

# CONST versus all classification.s
y_n = np.array(2*(abs(y_n - recog) <= 0.001)-1)

prob  = svm_problem(y_n, x_n)
param = svm_parameter('-s 0 -c 0.01 -t 1 -g 1 -r 1 -d 2 -v 10')
m = svm_train(prob, param)

# 0, Cross Validation Accuracy = 89.3705%
# 2, Cross Validation Accuracy = 89.9739%
# 4, Cross Validation Accuracy = 91.0575%
# 6, Cross Validation Accuracy = 90.8929%
# 8, Cross Validation Accuracy = 92.5662%

# 1, Cross Validation Accuracy = 98.5599%
# 3, Cross Validation Accuracy = 90.9752%
# 5, Cross Validation Accuracy = 92.3742%
# 7, Cross Validation Accuracy = 91.1535%
# 9, Cross Validation Accuracy = 91.1672%