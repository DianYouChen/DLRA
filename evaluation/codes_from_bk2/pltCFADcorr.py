#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:14:39 2020

@author: handsomedong
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

target_sites = ["Taipei", "Anbu", "Fushan"]
#==============================================================================
path = "/bk2/handsomedong/database/finalCFAD/"

def main():
    num = np.array([]) # count total data number
    for k in range(len(target_sites)):
        site = target_sites[k]
        target_file = path+site+"_2p5_combine.h5"
        # Load Data
        with h5py.File(target_file, 'r') as f:
            cfad = f['CFAD'][:]
            qc = f['qc'][:]
        rain = pd.read_hdf(target_file, keys='rain',mode='r')
        # QC
        def cfadQC(pd_rain, qc1, cfad1):
            rainfall = pd_rain.PP01.values
            loc, = np.where(rainfall > 0) # non-zero location
            outlier_index = [loc[x] for x in range(len(loc)) if qc1[loc][x] == 1]
            cfad1 = np.delete(cfad1, outlier_index, axis=0)
            rain1 = pd_rain.drop(index=outlier_index).reset_index(drop=True)
            qc1 = np.delete(qc1, outlier_index, axis=0)
            return rain1, qc1, cfad1
        rain, qc, cfad = cfadQC(rain, qc, cfad)
        
        cfad2 = np.sum(cfad, axis=1)/6/13
        rain2 = rain.PP01.values
        del qc, site, target_file
        # delete zero data
        def kickzero(rainfall, cfaddd):
            zeroMask = np.where(rainfall == 0)[0]
            rainfall = np.delete(rainfall, zeroMask, axis=0)
            cfaddd = np.delete(cfaddd, zeroMask, axis=0)
            return rainfall, cfaddd
        rain3, cfad3 = kickzero(rain2, cfad2)
        num = np.append(num, len(rain3))
        # main part
        if k == 0:
            corrcoef = np.full([len(target_sites),np.size(cfad3,1),np.size(cfad3,2)],np.nan)
        for i in range(np.size(corrcoef, 1)):
            for j in range(np.size(corrcoef, 2)):
                df = pd.DataFrame({'cfad_value': cfad3[:,i,j],
                                   'rain_value': rain3})
                corrcoef[k,i,j] = df.corr().iloc[0,1]
                del df
        del i, j
    return corrcoef, num
corrcoef, data_num = main()
#%%
import matplotlib
import sys
sys.path.append("/bk2/handsomedong/PythonLearning/analysis/")
from theFunction import shiftedColorMap

orig_cmap = matplotlib.cm.bwr
shifted_cmap = shiftedColorMap(orig_cmap,
                               start=0, 
                               midpoint=0.3, 
                               stop=1, 
                               name='shifted')
def plot():
    font = {'family'     : 'DejaVu Sans Mono',
            'weight'     : 'bold',
            'size'       : 14
            }
    axes = {'titlesize'  : 18,
            'titleweight': 'bold',
            'labelsize'  : 17,
            'labelweight': 'bold'
            }
    mpl.rc('font', **font)  # pass in the font dict as kwargs
    mpl.rc('axes', **axes)
    
    lv = np.append(np.arange(1,5.5,0.5),np.arange(6,18))
    xAxis = np.arange(-5,70,5)
    
    # plot
    fig, ax = plt.subplots(1, 3, num='corr', figsize=(15, 7), dpi=100, facecolor='w')
    fig.suptitle('Correlation Coefficient of each grid vs. rainfall',
                 x=0.5,
                 y=1.05,
                 fontsize=18, 
                 fontweight='heavy')
    plt.tight_layout(h_pad=2)
    for i in range(len(data_num)):
        pc = ax[i].pcolor(xAxis, lv, corrcoef[i,:,:], cmap=shifted_cmap, vmax=.7, vmin=-.3)
        ax[i].text(-4, 16.1, 'Total data Number='+str(int(data_num[i])),fontsize=13)
        ax[i].set_xticks(xAxis)
        ax[i].set_xticklabels(['Neg',0,5,10,15,20,25,30,35,40,45,50,55,60])
        ax[i].set_yticks(np.arange(1,18))
        ax[i].grid(b=True, which='major', axis='both', c='gray', ls='-.')
        ax[i].set_title(target_sites[i])
    ax[0].set_ylabel('Height(km)', fontsize=16, fontweight='bold')
    cbar = fig.colorbar(pc,
                 ax=ax[:],
                 orientation='horizontal',
                 shrink=0.5,
                 aspect=40,
                 extend='both',)
    ax[1].set_xlabel('dBZ', fontsize=17, fontweight='bold')
#    fig.savefig('/bk2/handsomedong/selfmade/corrcoef/corrMap1.png', dpi=100)
    #plt.close(fig)

plot()

#%% Single One
import matplotlib
import sys
sys.path.append("/bk2/handsomedong/PythonLearning/analysis/")
from theFunction import shiftedColorMap

orig_cmap = matplotlib.cm.bwr
shifted_cmap = shiftedColorMap(orig_cmap,
                               start=0, 
                               midpoint=0.3, 
                               stop=1, 
                               name='shifted')
def plot():
    font = {'family'     : 'DejaVu Sans Mono',
            'weight'     : 'bold',
            'size'       : 14
            }
    axes = {'titlesize'  : 18,
            'titleweight': 'bold',
            'labelsize'  : 16,
            'labelweight': 'light'
            }
    mpl.rc('font', **font)  # pass in the font dict as kwargs
    mpl.rc('axes', **axes)
    
    lv = np.append(np.arange(1,5.5,0.5),np.arange(6,18))
    xAxis = np.arange(-5,70,5)
    
    # plot
    fig, ax = plt.subplots(1, 1, num='corr', figsize=(8,6), dpi=100, facecolor='w')
#    fig.suptitle('Correlation Coefficient of each grid VS. rainfall',
#                 x=0.5,
#                 y=0.92,
#                 fontsize=15, 
#                 fontweight='heavy')
    plt.tight_layout()
    i=2
    pc = ax.pcolor(xAxis, lv, corrcoef[i,:,:], cmap=shifted_cmap, vmax=.7, vmin=-.3)
    ax.text(-4, 16.1, 'Total data Number='+str(int(data_num[i])),fontsize=14)
    ax.set_xticks(xAxis)
    ax.set_xticklabels(['Neg',0,5,10,15,20,25,30,35,40,45,50,55,60])
    ax.set_yticks(np.arange(1,18))
    ax.grid(b=True, which='major', axis='both', c='gray', ls='-.')
    ax.set_title(target_sites[i])
    ax.set_ylabel('Height(km)', fontsize=16, fontweight='bold')
    cbar = fig.colorbar(pc,
                 ax=ax,
                 orientation='vertical',
                 shrink=0.85,
                 aspect=40,
                 extend='both',)
    ax.set_xlabel('dBZ', fontsize=16, fontweight='bold')
    #fig.savefig('/bk2/handsomedong/selfmade/corrcoef/corrMap1.png', dpi=100)
    #plt.close(fig)

plot()
