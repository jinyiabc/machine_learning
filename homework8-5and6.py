#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:13:10 2019

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
param = svm_parameter('-s 0 -c 1 -t 1 -g 1 -r 1 -d 2')
m = svm_train(prob, param)

# 0.001 Total nSV = 72 Cross Validation Accuracy = 99.5516%
# 0.01 Total nSV = 32 Cross Validation Accuracy = 99.4875%
# 0.1 Total nSV = 24 Cross Validation Accuracy = 99.4875%
# 1 Total nSV = 22 Cross Validation Accuracy = 99.5516%

#df2 = df2.loc[df2['digit'].isin(array)]
#
#x1 = df2['intensity']
#x2 = df2['symmetry']
#x_n = np.column_stack((x1, x2))
#y_n = df2['digit']
#
## CONST versus all classification.
#y_n = np.array(2*(abs(y_n - recog) <= 0.001)-1)

# E_IN err / E_OUT err for df2
p_labels, p_acc, p_vals = svm_predict(y_n, x_n, m)

# 0.001 
#Total nSV = 76
#Accuracy = 98.3491% (417/424) (classification)

#0.01
#Total nSV = 34
#Accuracy = 98.1132% (416/424) (classification)

# 0.1
#Total nSV = 24
#Accuracy = 98.1132% (416/424) (classification)

# 1
#Total nSV = 24
#Accuracy = 98.1132% (416/424) (classification)

# q = 2 c = 0.01
# Total nSV = 34
#Accuracy = 99.5516% (1554/1561) (classification)

#q = 5 c = 0.01
#Total nSV = 23
#Accuracy = 99.6156% (1555/1561) (classification)

#q = 5, c=0.0001
#Total nSV = 26
#Accuracy = 99.5516% (1554/1561) (classification)

#q = 2, c=0.0001
#Total nSV = 236
#Accuracy = 99.1031% (1547/1561) (classification)

#c=0.001 q = 2
#Total nSV = 76

#c= 0.001 q=5
#Total nSV = 25

#c=1 q=5
#Accuracy = 99.6797% (1556/1561) (classification)

#c=1, q=2
#Accuracy = 99.6797% (1556/1561) (classification)