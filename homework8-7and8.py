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
range_c = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])
m = np.zeros(5)
sum = np.zeros(5)
for i in range(100):
    for idx, value in enumerate(range_c):
        para = '-s 0 -c ' + str(value) + ' -t 1 -g 1 -r 1 -d 2 -v 10 -q'
        param = svm_parameter(para)
        m[idx] = svm_train(prob, param)
        sum[idx] = sum[idx] + m[idx]
print(sum/100)

#[99.50992953 99.52914798 99.52850737 99.51761691 99.02818706]
#[99.5169763  99.52210122 99.53299167 99.51825753 99.03523382]
#[99.52017937 99.52081999 99.53171044 99.52658552 99.03523382]