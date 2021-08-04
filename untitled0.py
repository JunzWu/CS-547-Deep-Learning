# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:58:05 2019

@author: Junz
"""

import numpy as np
a = np.load('single_frame_confusion_matrix.npy')
for i in range(101):
    print(a[i,:].sum())

b = a.reshape(1,-1)
c = b.argsort()
c = c[0, -10:]
d = []
for i in range(10):
    x = c[i]//101
    y = c[i]%101
    d.append([x,y])
print(d)