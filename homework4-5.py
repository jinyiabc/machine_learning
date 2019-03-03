#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:47:52 2019

@author: jinyi
"""

import numpy as np

coeff = np.array(([1.43779027]))

# y = c + mx
qty = 100
x0 = np.arange(-1, 1, 2.0/(qty*1.0), dtype=float)
# x = np.column_stack((np.ones(qty), x0))
y_g = x0*coeff
y_f = np.sin(x0*np.pi)
dif = (y_g - y_f)**2
Exp_variance = np.mean(dif)
print(Exp_variance)