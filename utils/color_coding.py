# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 08:32:02 2023

@author: jakob
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np


fig = plt.gcf()
fig.set_size_inches(20, 12)
num_plots = 25

divider = 5
c1 = pl.cm.Blues(np.linspace(0.2,1,divider))
c2 = pl.cm.Greens(np.linspace(0.2,1,divider))
c3 = pl.cm.Greys(np.linspace(0.2,1,divider))
c4 = pl.cm.Purples(np.linspace(0.2,1,divider))
c5 = pl.cm.Reds(np.linspace(0.2,1,divider))

colors = [c1, c2, c3, c4, c5]

seq_count = 0
wrap_count = 0

for i in range(num_plots):
    x = [x for x in range(num_plots)]
    y = [y*i for y in range(num_plots)]
    plt.plot(x,y, label = f"{i}", color=colors[wrap_count][seq_count])
    seq_count+=1
    if seq_count == divider:
        wrap_count+=1 
        seq_count=0
plt.legend()
plt.show()

    