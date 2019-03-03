#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:56:42 2019

@author: jinyi
"""

import numpy as np
p = np.poly1d([1, 2, 1])
p1 = np.polyder(p)
solution = np.roots(p)
