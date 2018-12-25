#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<Description>

Created on Thu Dec 13 16:06:51 2018
@author: Richard Boyne rmb115@ic.ac.uk
"""

import numpy as np
import pandas as pd
from sph_fe import sph_simulation


def test_short_run():
    """
    This runs a small simulation to check all the moving parts are working
    """
    def f(x, y):
        if 0 <= y <= 2 or (0 <= x <= 3 and 0 <= y <= 5):
            return 1
        else:
            return 0

    sph_simulation(x_min=[0, 0], x_max=[20, 10], t_final=1, dx=0.8,
                   func=f, ani=False, file_name="to_travis")


def test_speedofsound():
    "test the speed never exceeds speed of sound (20)"
    # load the data
    file_name = 'to_travis.csv'
    data = pd.read_csv(file_name, skiprows=2, index_col=False)
    data = data.set_index('Time')

    # run the test
    v = np.sqrt(data['V_x']**2+data['V_y']**2)
    assert np.all(v < 20)


def test_density():
    # load the data
    file_name = 'to_travis.csv'
    data = pd.read_csv(file_name, skiprows=2, index_col=False)
    data = data.set_index('Time')
    times = np.unique(data.index)

    dens = data['Density']
    rho0 = dens.loc[times[0]].mean()

    # find the data lims
    N = len(times)-1  # ignore the first time
    minimums = np.empty(N)
    maximums = np.empty(N)
    stds = np.empty(N)
    for i, t in enumerate(times[1:]):
        minimums[i] = dens.loc[t].min()
        maximums[i] = dens.loc[t].max()
        stds[i] = dens.loc[t].std()

    # assert these values
    assert np.all(minimums > 0)  # check that density is positive
    assert np.all(minimums > rho0/1.5)  # check density is in bounds
    assert np.all(maximums < rho0*1.5)  # check density is in bounds
    assert np.all(stds > 0)  # check that density changes


def test_overlap():
    """
    Can only test overlap on a couple of points due to computation time
    """
    # load the data
    file_name = 'to_travis.csv'
    data = pd.read_csv(file_name, skiprows=2, index_col=False)
    data = data.set_index('Time')
    times = np.unique(data.index)

    for _ in range(2):
        t = times[np.random.randint(len(times))]
        current = data.loc[t]
        positions = np.array(list(zip(current['R_x'], current['R_y'])))

        for i, p in enumerate(positions):
            for j, other in enumerate(positions):
                if i != j:
                    diff = sum((p-other)**2)
                    assert not np.isclose(diff, 0)
