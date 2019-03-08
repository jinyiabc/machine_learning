#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:04:41 2019

@author: jinyi
"""

import numpy as np
from numpy import minimum as min
sum = 0

for i in np.arange(1000):
    e1, e2 = np.random.uniform(low=0.0, high=1.0, size=2)
    e3 = min(e1, e2)
    sum += e3
print("E(min(e1,e2))=", sum/1000)

# E(min(e1,e2))= 0.3328519922624037
