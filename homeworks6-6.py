

import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import transpose, matmul, dot

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
# A*w = y
#E_IN = np.count_nonzero(np.sign(np.dot(A, w)) != y)/35.0
#print(E_IN)
#k = -1
    
def err(z, y, w, k):
    result = np.dot(np.transpose(np.dot(z, w) - y), (np.dot(z, w) - y))/(y.size)
    lamda = 10**k
    #lamda = 0
    result+=lamda/y.size*np.dot(w, w)
    return result
    
def der(z, y, w):
    result = np.dot(np.transpose(z), (np.dot(z, w) - y))/(y.size)
    return result 

k = 1
cur_w = np.linalg.lstsq(A, y, rcond=None)[0]
pre_w = np.zeros(8)
#cur_w = np.zeros(8)
#print("initialization E_IN", err(A, y, cur_w, k))
eta = 0.01
precision = 10**(-7)
N = y.size
lamda = 10**(-k)
max_iters = 10000
itr = 0

# Slides 12: 11/21
temp = inv(dot(transpose(A), A) + lamda*np.identity(8))
cur_w = matmul(matmul(temp, transpose(A)), y)



E_IN = np.count_nonzero(np.sign(np.dot(A, cur_w)) != y)/35.0
print("E_IN:", E_IN)  
#print("regression err:", err(A, y, cur_w, -k))  
print("k:", -k, "iteration:", itr)

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
E_OUT = np.count_nonzero(np.sign(np.dot(A, cur_w)) != y)/250.0
print("E_OUT:", E_OUT)


#E_IN: 0.02857142857142857
#k: -1 iteration: 0
#E_OUT: 0.056