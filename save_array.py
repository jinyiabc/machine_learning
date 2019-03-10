#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:41:14 2019

@author: jinyi
"""

#from tempfile import TemporaryFile
#outfile = TemporaryFile()
import numpy as np

x = np.arange(10)
#np.save(outfile, x)

#outfile.seek(0) 
#np.load(outfile)

np.save('test3.npy', x) 
d = np.load('test3.npy')
print(x == d)