#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:26:16 2019

@author: jinyi
"""

import numpy as np
from numpy import sqrt
precision = 0.1
# if h0(x) = b
E_cv = (1 + 0.5 +0.5)/3

range = np.array([sqrt(sqrt(3) + 4), sqrt(sqrt(3) - 1), sqrt(9 + 4*sqrt(6)), sqrt(9 - sqrt(6))])
# if h0(x) = ax + b
for x in range:

    e1 = abs(np.interp(-1, [1, (x)], [0, 1]) - 0)  # (-1, 0)
    e2 = abs(np.interp(1, [-1, (x)], [0, 1]) - 0)  # (1, 0)
    e3 = 1  # (x, 1)
    E_cv1 = (e1 + e2 + e3)/3
    print(E_cv1)
#    if abs(E_cv1 - E_cv) <= precision:
#        break
# print(x)