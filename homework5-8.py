# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:34:57 2019

@author: yijin
"""

import numpy as np
# machine limits epsilon
eps = np.finfo(float).eps 
def gradient_descent(w, x, y):
    "Logistic regression: e(w) = ln(1+e**(-y*wT*x)"
    err0 = -y*x[0]/(1 + np.exp(y*np.dot(w, x)))
    err1 = -y*x[1]/(1 + np.exp(y*np.dot(w, x)))
    err2 = -y*x[2]/(1 + np.exp(y*np.dot(w, x)))
    return np.array([err0, err1, err2])

  
data_set = np.random.uniform(low=-1, high=1.0+eps, size=(2,2)) 
# y = m*x + c
A = np.column_stack((data_set[:,0], np.ones(2)))
y = data_set[:,1]
m, c = np.linalg.lstsq(A, y, rcond=None)[0]

w0 = np.array([m, -1, c])

sample = np.random.uniform(low=-1, high=1.0+eps, size=(100,2))
X = np.column_stack((sample, np.ones(100)))
Y = np.sign(np.dot(X, w0))
cur_w = np.zeros(3)
pre_w = np.ones(3)
precison = 0.01
iter = 0
while np.linalg.norm((pre_w - cur_w)) >= precison:
    pre_w = cur_w
    shuffle = np.random.choice(100, 100, replace=False)
    
    eta = 0.01
    for i in shuffle:
        GD = gradient_descent(pre_w, X[i], Y[i])    
        cur_w = pre_w - eta*GD
        #print(cur_w)
    iter +=1
print(cur_w)    
A0 = np.column_stack((data_set, np.ones(2)))
print("test:",np.dot(A0, cur_w),np.dot(A0, w0))
print("precision", np.linalg.norm((pre_w - cur_w))) 
print("iteration:", iter)   

def err(w, x, y):
    "cross entropy"
    result = np.log((1+np.exp(-y*np.dot(w, x))))
    return result

N1 = 100
out_of_sample = np.random.uniform(low=-1, high=1.0+eps, size=(N1, 2))    
#X2 = out_of_sample[:,0]
#Y2 = out_of_sample[:,1]    
#YY = np.sign(Y2 - m*X2 - c)

XX = np.column_stack((out_of_sample, np.ones(N1)))
YY = np.sign(np.dot(XX, w0))
sum_err = 0
for j in range(N1):
    sum_err = sum_err + err(cur_w, XX[j], YY[j])
print("E_out:", sum_err/N1)
