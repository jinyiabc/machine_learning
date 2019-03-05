#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:43:16 2019

@author: jinyi
"""

import pandas as pd

df = pd.read_stata('http://www.principlesofeconometrics.com/stata/mroz.dta')
print(df.head())

df1 = pd.read_csv('in.csv')