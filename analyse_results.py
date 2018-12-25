#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<Description>

Created on Thu Dec 13 16:07:17 2018
@author: Richard Boyne rmb115@ic.ac.uk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# set matplotlib preferences - viewing
plt.rc('axes', titlesize=20, labelsize=20)
plt.rc('axes.formatter', limits=[-4, 4])
plt.rc('ytick', labelsize=12)
plt.rc('xtick', labelsize=12)
plt.rc('lines', linewidth=1.5, markersize=7)
plt.rc('figure', figsize=(9, 9))
plt.rc('legend', fontsize=15)

#data = pd.read_csv('raw_data/deb-overnight-small.csv', skiprows=2,
#                   index_col=False).set_index('Time')
times = np.unique(data.index)

# plot the density values with time
if 1:
    dens = data['Density']
    rho0 = dens.loc[0].mean()
    
    # find the data lims
    minimum = []
    maximum = []
    average = []
    std = []
    for t in times:
        minimum.append(dens.loc[t].min())
        maximum.append(dens.loc[t].max())
        average.append(dens.loc[t].mean())
        std.append(dens.loc[t].std())

    # plot the results
    fig, ax = plt.subplots()
    ax.plot(times[::10], minimum[::10], 'r--', label='min')
    ax.plot(times[::10], maximum[::10], 'g-', label='max')
    ax.plot(times[::10], average[::10], 'b--', label='avg')
    ax.plot(times[::10], np.array(average[::10]) + np.array(std[::10]),
            'b-', label='upper std')
    ax.plot(times[::10], np.array(average[::10]) - np.array(std[::10]),
            'r-', label='lower std')
    
#    ax.hlines(rho0*1.5, min(times), max(times), color='k', label=r'$1.5\rho_0$')
#    ax.hlines(rho0*0.5, min(times), max(times), label=r'$0.5\rho_0$')
    
    ax.set(title='Density vairations with time', xlabel=r'Time $[s]$',
           ylabel=r'Density $[\frac{kg}{m^3}]$')
    ax.legend()