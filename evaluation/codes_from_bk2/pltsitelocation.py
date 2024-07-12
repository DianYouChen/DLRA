#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:49:55 2020

@author: handsomedong
"""

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.extend(['/bk2/handsomedong/PythonLearning/analysis/',])
from cmap_terrain import cmap_height

mat = sio.loadmat('/bk2/handsomedong/database/data/terrain_TW.mat')
terrain_Taiwan = mat['Taiwan']
terrain_lat = np.linspace(26.0001, 20.9999, 18001)
terrain_lon = np.linspace(118.9999, 123.0001, 14401)
del mat
mat = sio.loadmat('/bk2/handsomedong/database/data/city_lonlat_region5.mat')
citylon = mat['citylon']
citylat = mat['citylat']
del mat
site_list = pd.read_excel('/bk2/handsomedong/database/data/NorthTW_StationSite.xlsx',
                          header=None,
                          index_col=0,
                          names=['SiteCode', 'Lon', 'Lat'])
Outlier_station = ['Wugu','Wufengshan','Songshan','Siapen','Quchi', 'Linkou',
                   'FBG', 'Baishawan', 'Pingdeng', 'Qidu','Shenkeng', 
                   'Sanchong', 'Datunshan']
site_list2 = site_list.drop(index=Outlier_station)

#%% plot Terrain
def set_font():
    import matplotlib as mpl
    font = {'family'     : 'DejaVu Sans Mono',
            'weight'     : 'bold',
            'size'       : 14
            }
    axes = {'titlesize'  : 18,
            'titleweight': 'bold',
            'labelsize'  : 16,
            'labelweight': 'bold'
            }
    mpl.rc('font', **font)  # pass in the font dict as kwargs
    mpl.rc('axes', **axes)
set_font()

fig, ax = plt.subplots(num='terrain', figsize=(10, 7.5), dpi=100, facecolor='w')
#cf = ax.contourf(terrain_lon, terrain_lat, terrain_Taiwan, cmap=cmap_height(),levels=np.arange(0,2000,400))
#cf = ax.contourf(terrain_lon, terrain_lat, terrain_Taiwan, cmap=cmap_height(),vmin=0,vmax=2000)
ax.imshow(terrain_Taiwan,
          cmap=cmap_height(),
          extent=[terrain_lon[0],terrain_lon[-1],terrain_lat[-1],terrain_lat[0]],
          vmax=1000,vmin=0)
ax.grid(axis='both', ls='-')
ax.set_title('Northern Station Sites and Terrain')
### set colorbar not figure
m = plt.cm.ScalarMappable(cmap=cmap_height())
m.set_array(terrain_Taiwan)
m.set_clim(0,1000)
cbar = fig.colorbar(m, shrink=0.7)
cbar.ax.set_title('m')
###
ax.plot(citylon,citylat,'k',linewidth=2)
ax.axis([121.2, 122.2, 24.6, 25.4]) # whole Taiwan # whole area [119, 123, 21, 26]
ax.set_aspect('equal')
###
pt1, = ax.plot(site_list2.Lon,site_list2.Lat,'r1',ms=10,mew=3)
pt2, = ax.plot(site_list.loc[Outlier_station].Lon,
               site_list.loc[Outlier_station].Lat,
               'b1',ms=10,mew=3)
ax.legend(handles=[pt1, pt2], 
          labels=['used(44)', 'not used(13)'], 
          loc='upper right',
          fontsize=14)
#%% magic power
import os
import matplotlib as mpl
path = "/bk2/handsomedong/database/finalSaver/"
files = os.listdir(path)
cnn_files = [path+x for x in files if x.endswith("CNN_v2.h5")]
cnn_files.sort()
del files, path
sitename=[]
for i in range(len(cnn_files)):
    #
    loc = cnn_files[i].find("_")
    sitename.append(cnn_files[i][38:loc])
del i, loc, cnn_files
# Load prepared data
rmse_cnn_10 = np.load("/bk2/handsomedong/database/data/rmse_cnn_10.npy")
rmse_zrr_10 = np.load("/bk2/handsomedong/database/data/rmse_zrr_10.npy")
#progress = np.where((rmse_zrr_10-rmse_cnn_10)>-1)[0]
#not_progress = np.array(sitename)[(rmse_zrr_10-rmse_cnn_10)<-1].tolist()
mean = pd.Series(rmse_zrr_10-rmse_cnn_10).mean()
std = pd.Series(rmse_zrr_10-rmse_cnn_10).std()
progress = np.where((rmse_zrr_10-rmse_cnn_10)>(mean-std))[0]
not_progress = np.array(sitename)[(rmse_zrr_10-rmse_cnn_10)<(mean-std)].tolist()

def set_font():
    font = {'family'     : 'DejaVu Sans Mono',
            'weight'     : 'bold',
            'size'       : 10
            }
    axes = {'titlesize'  : 15,
            'titleweight': 'bold',
            'labelsize'  : 15,
            'labelweight': 'bold'
            }
    mpl.rc('font', **font)  # pass in the font dict as kwargs
    mpl.rc('axes', **axes)
set_font()


fig, ax = plt.subplots(num='terrain', figsize=(10, 7.5), dpi=100, facecolor='w')
ax.imshow(terrain_Taiwan,
          cmap=cmap_height(),
          extent=[terrain_lon[0],terrain_lon[-1],terrain_lat[-1],terrain_lat[0]],
          vmax=1000,vmin=0)
ax.grid(axis='both', ls='-')
ax.set_title('Northern Station Sites and Terrain')
### set colorbar not figure
m = plt.cm.ScalarMappable(cmap=cmap_height())
m.set_array(terrain_Taiwan)
m.set_clim(0,1000)
cbar = fig.colorbar(m, shrink=0.7)
cbar.ax.set_title('m')
###
ax.plot(citylon,citylat,'k',linewidth=2)
ax.axis([121.2, 122.2, 24.6, 25.4])# whole area [119, 123, 21, 26]
ax.set_aspect('equal')
### add site
pt1, = ax.plot(site_list.iloc[progress, 1],
               site_list.iloc[progress, 2],
               'r1',ms=10,mew=2)
pt2, = ax.plot(site_list.loc[not_progress].Lon,
               site_list.loc[not_progress].Lat,
               'b1',ms=10,mew=2)
ax.legend(handles=[pt1, pt2], 
          labels=['Progress or steady', 'Regress'], 
          loc='upper right',
          fontsize=12)




