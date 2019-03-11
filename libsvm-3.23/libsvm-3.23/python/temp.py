#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:16:25 2019

@author: jinyi
"""

from svmutil import *
# Read data in LIBSVM format
y, x = svm_read_problem('../heart_scale')
m = svm_train(y[:200], x[:200], '-c 4')
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)

# Construct problem in python format
# Dense data
y, x = [1,-1], [[1,0,1], [-1,0,-1]]
# Sparse data
y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
prob  = svm_problem(y, x)
param = svm_parameter('-t 0 -c 4 -b 1')
m = svm_train(prob, param)

# Precomputed kernel data (-t 4)
# Dense data
y, x = [1,-1], [[1, 2, -2], [2, -2, 2]]
# Sparse data
y, x = [1,-1], [{0:1, 1:2, 2:-2}, {0:2, 1:-2, 2:2}]
# isKernel=True must be set for precomputed kernel
prob  = svm_problem(y, x, isKernel=True)
#param = svm_parameter('-t 4 -c 4 -b 1')
param = svm_parameter('-s 0 -c 0.01 -t 1 -g 1 -r 1 -d 2')
m = svm_train(prob, param)

#p_labels, p_acc, p_vals = svm_predict(y, x, m)
