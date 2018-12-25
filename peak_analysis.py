# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:10:21 2018

@author: karad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read files


def Read(file_name):
    
    # read from files
    df = pd.read_csv(file_name, skiprows = 2, index_col=False)

    # format data
    tt = df['Time']
    tt = [float(i) for i in tt]
    xx = df['R_x']
    xx = [float(i) for i in xx]
    yy = df['R_y']
    yy = [float(i) for i in yy]
    boundary = df['Boundary']
    boundary = [int(i) for i in boundary]
    
    
    # Remove Boundary particles
    x = []
    y = []
    t = []
    for i in range(len(boundary)):
        if boundary[i] == False:
            x.append(xx[i]);
            y.append(yy[i]);
            t.append(tt[i])
    
    x = np.array(x); y=np.array(y); t=np.array(t)

    return  x, y, t



def t_index(t):
    """
    input: a list
    Return: a list with coordiates where list[i] != list[i]
    """
    dt = np.diff(t)
    store=[]
    store.append(0)
    for i in range(len(dt)):
        
        # coordinate when t changes value
        coordinate=i+1
        if dt[i] != 0:
            store.append(coordinate)
    
    # last particle coordinate
    store.append(len(dt)+1)
    
    return store





def MVAVARAGE(List, N):
    """
    input:
        List- a list
        N- N point avarage
    Returns:
        N point avarage of the List
    
    """
    cumsum, moving_aves = [0], []
    
    for i, x in enumerate(List, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
        else:
            moving_ave = 0
        moving_aves.append(moving_ave)
    return(moving_aves)


def peak(file_name, mv_avN, wallpos):
    """
    Inputs: 
        ----------------------------------
        x - x_coordinates array (exclude walls)
        y - y_coordinates array (exclude walls)
        t - t_coordinates array (exclude walls)
        mv_avN - N move avarage (float or int)
        wallpos - x-coordinates of boundary (float or int)
        ---------------------------------
    Outputs:
        ---------------------------------
        Crest_xcoords (list), Crest_height (list), times(list), sloash time(flaot)
        ---------------------------------
    """
    x, y, t =  Read(file_name)

    
    x=np.array(x)
    y=np.array(y)
    t=np.array(t)

    # time values without repetition
    ts_unsorted= list(set(t))
    ts = np.sort(ts_unsorted)
    
    # tindex = start index of each time
    tindex = t_index(t)
    #print(tindex)

    Crest_height=[]
    Crest_xcoords=[]
    t_sloshing=[]
    for i in range(len(ts)):
        
        # Slice x-array, y-array to get a timeframe.
        gridpointsX = x[tindex[i] : tindex[i+1]]
        ypoints = y[tindex[i] : tindex[i+1]]
        gridpointsY = MVAVARAGE(ypoints, mv_avN)
        
        Peak_Indx = np.where(gridpointsY==max(gridpointsY))[0][0]
        Peak = gridpointsY[Peak_Indx]
        Crest_height.append(Peak)
        coord = gridpointsX[Peak_Indx]
        Crest_xcoords.append(coord)
        
        
        if coord >= 0.85*wallpos:
            t_sloshing.append(ts[i])
            
        #print(tindex[i])
        #print(gridpointsX)
        #print(gridpointsY)
        #print(t_sloshing)
        #print(max(gridpointsY), Peak_Indx, coord)
    #plt.plot(ts, Crest_xcoords, 'o')
       
    tslosh = min(t_sloshing)
    
    #print("Crest_height", Crest_height)
    #print("Crest_xcoords", len(Crest_xcoords))
    #print("tslosh", tslosh)
    #print("ts", ts)
    #print(coord)
    
    return(Crest_xcoords, Crest_height, ts, tslosh)


Cx, Cy, ts, tslosh = peak('./raw_data/2018-12-12-23hr-39m.csv', 1, 20)

np.savetxt("Cx_File.csv", Cx, delimiter=",")
np.savetxt("Cy_File.csv", Cy, delimiter=",")
np.savetxt("ts_File.csv", ts, delimiter=",")
print(tslosh)
