#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:14:16 2019

@author: jinyi
"""

import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from numpy.linalg import inv
from numpy import transpose, matmul, dot
N = 7291
C = 0.01
Q = 2
const = 4
df1 = pd.read_csv('train.csv')
#df2 = pd.read_csv('test.csv')

x1 = df1['intensity']
x2 = df1['symmetry']
x_n = np.column_stack((x1, x2))
y_n = df1['digit']

# CONST versus all classification.s
y_n = transpose(np.column_stack(2*(abs(y_n - const) <= 0.001)-1))


def qp(x, y):
    # QP solution
    path = '/Volumes/SSD/p' + str(x) + '.npy'
    P = matrix(np.load(path), tc='d')
    q = matrix(-np.ones(N), tc='d')
    A = matrix(np.transpose(y), (1,N), tc='d')
    b = matrix(np.zeros(1), tc='d')
    G = matrix(np.load('/Volumes/SSD/g_array.npy'), tc='d')
    h = matrix(np.load('/Volumes/SSD/h_array.npy'), tc='d')
    
        #G1 = np.identity(N)
        #G2 = -np.identity(N)
        #G = matrix(np.vstack((G1, G2)), tc='d')
        #h1 = np.ones(N)*C
        #h2 = np.zeros(N)
        #h = matrix(np.hstack((h1, h2)), tc='d')
    
        #np.save('/Volumes/SSD/g_array.npy', np.vstack((G1, G2))) 
        #np.save('/Volumes/SSD/h_array.npy', h) 
    
    
    sol = solvers.qp(P,q,G,h,A,b)
    a = sol['x']
    ## Write solution to a#.npy.
    np.save('/Volumes/SSD/a' + str(x) + '.npy', a)
    return

qp(const, y_n)
    
path = '/Volumes/SSD/a' + str(const) + '.npy'
a = np.load(path)
precision = 10**(-3)

a0 = np.zeros(N)
count0 = 0
for i in range(N):
    if abs(a[i]) <= precision:
        a0[i] = 0
        count0 =count0 +1
    else:
        a0[i] = a[i]
#t = (1 + dot(x_n[i], x_n[j]))**Q
print(np.count_nonzero(a0))

    

#w = np.zeros((N,2))
#for i in range(N):
#    for j in range(2):
#        w[i][j] = a[i]*y_n[i]*x_n[i][j]

idx = np.argmax(a)
#idx1 = np.argmin(a)
# yn(wT*xn + b) = 1        
def wz(a, y, x, x1, Q):
    sum = 0
    for i in range(a.size):
        t = (1 + dot(x1, x[i]))**Q
        sum = sum + a[i]*y[i]*t
    return sum

b = y_n[idx] - wz(a0, y_n, x_n, x_n[idx], Q)

#sum = 0
#for i in range(N):
#    if y_n[i] != np.sign(wz(a0, y_n, x_n, x_n[i], Q) + b):
#        sum = sum + 1
#print(sum/N)



## 2 versus all E_ in:                           18
## 4 versus all E_in: 0.8192291866684954         
## 6 versus all E_in: 0.6985324372514058         3347
## 8 versus all E_in  0.925661774790838           32
## 0 versus all E_in: 0.8362364559045399      sv: 40



