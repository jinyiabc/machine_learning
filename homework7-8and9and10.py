# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:18:40 2019

@author: yijin
"""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
# machine limits epsilon
eps = np.finfo(float).eps 
N = 100
N1 = 1000
sum_err = 0
sum_iter = 0
for i in range(1000):
  
    data_set = np.random.uniform(low=-1, high=1.0+eps, size=(2,2)) 
    # y = m*x + c
    A = np.column_stack((data_set[:,0], np.ones(2)))
    y = data_set[:,1]
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    w_0 = np.array([c, m, -1])
    w_00 = [1, w_0[1]/w_0[0], w_0[2]/w_0[0]]
    
    data_set = np.random.uniform(low=-1, high=1+eps, size=(N,2))
    x_set = np.column_stack((np.ones(N),data_set[:,0],data_set[:,1]))
    y_set = np.sign(np.matmul(x_set,w_0))
    
    w = np.random.uniform(low=-1, high=1+eps, size=3)
    
    def optimize(x,y,w):
        "test"
        y_1 = np.sign(np.matmul(x,w))        
#        print(y_1)
        y_2 = 2*(y_1 == y)-1            
        for idx, value in enumerate(y_2):
            if value == -1:
                w = w + x[idx]*y[idx]
                return w
    
    for i in range(500):
        test = (np.sign(np.matmul(x_set,w)) == y_set)
        if np.all(test):
#            sum += i+1.0 
            break
        else:
            w = optimize(x_set,y_set,w)
            continue
    w_11 = [1, w[1]/w[0], w[2]/w[0]]    
    #w_11 = w
    
    p1 = np.identity(3)
    p1[0][0] = 0.0
    P = matrix(p1, tc='d')
    q = matrix(np.zeros(3), tc='d')
    #  -y_n*T(x_n)* w - y_n*b <= -1
    # G = -y_n*T(x_n)
    # h = -1 
    x_set1 = np.zeros((N,3))
    for idx, value in enumerate(y_set):
        x_set1[idx] = x_set[idx]*value*(-1)
        
        
    G = matrix(np.array(x_set1), tc='d')
    h = matrix(np.ones(N)*(-1), tc='d')
    sol = solvers.qp(P,q,G,h)
    
    w_2 = sol['x']
    a, b, c = w_2
    w_22 = [1, b/a, c/a]
    #sol['primal objective']
    precision = 0.0001
    iter = 0
    #test = np.zeros(N)
    for i in range(N):
        #test[i] = (abs(np.dot(x_set1[i], w_2) + 1) <= precision)
        if (abs(np.dot(x_set1[i], w_2) + 1) <= precision):
            iter = iter +1
    sum_iter = sum_iter + iter    
    
    
    #N1 = 10000
    out_of_sample = np.random.uniform(low=-1, high=1.0+eps, size=(N1, 2))    
    XX = np.column_stack((np.ones(N1), out_of_sample))
    F = np.sign(np.matmul(XX, w_00))
    G_PLA = np.sign(np.matmul(XX, w_11))
    G_SVM = np.sign(np.matmul(XX, w_22))
    err_pla = np.count_nonzero(F != G_PLA)/N1
    err_svm = np.count_nonzero(F != G_SVM)/N1
    sum_err = sum_err + (err_pla > err_svm)
    print("PLA:", err_pla)
    print("SVM", err_svm)

print("How often svm is better than pla:", sum_err/1000)
print("Average number of support vectors:", sum_iter/1000)

