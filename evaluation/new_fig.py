import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import torch
from torch import nn

font = {'family'     : 'sans-serif',
        'weight'     : 'bold',
        'size'       : 14
        }
axes = {'titlesize'  : 16,
        'titleweight': 'bold',
        'labelsize'  : 14,
        'labelweight': 'bold'
        }
mpl.rc('font', **font)  # pass in the font dict as kwargs
mpl.rc('axes', **axes)

#set colorbar
cwbRR = mpl.colors.ListedColormap(['#FFFFFF', '#9CFCFF', '#03C8FF', '#059BFF', '#0363FF',
                                   '#059902', '#39FF03', '#FFFB03', '#FFC800', '#FF9500',
                                   '#FF0000', '#CC0000', '#990000', '#960099', '#C900CC',
                                   '#FB00FF', '#FDC9FF'])
bounds = [ 0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
norm = mpl.colors.BoundaryNorm(bounds, cwbRR.N)

def plotLongFig(inp): #在funct中的變數執行完就會被清除
    t_len = inp.shape[0]
    item = inp.shape[1]
    ### city edge
    mat = sio.loadmat('city_lonlat_region5.mat')
    citylon = mat['citylon']
    citylat = mat['citylat']
    del mat
    
    ### model axis
    latStart = 20; latEnd = 27;
    lonStart = 118; lonEnd = 123.5;
    lat = np.linspace(latStart,latEnd,561)
    lon = np.linspace(lonStart,lonEnd,441)
    lon, lat = np.meshgrid(lon[215:335], lat[325:445])
    
    fig, ax = plt.subplots(item, t_len, figsize=(10, 8.5), dpi=200, facecolor='w')
    for time_step in range(t_len):
        for y in range(item): # gt, 0-1, 1-2, 2-3
            ax[y,time_step].plot(citylon,citylat,'k',linewidth=0.6)
            ax[y,time_step].axis([120.6875, 122.1875, 24.0625, 25.5625])# whole area [119, 123, 21, 26]
            ax[y,time_step].set_aspect('equal')
            ax[y,time_step].pcolormesh(lon, lat, inp[time_step, y], edgecolors='none',
                                    shading='auto', norm=norm, cmap=cwbRR)
            ax[y,time_step].set_xticks([])
            ax[y,time_step].set_yticks([])
            
def cal_interval_rmse(data, target, thsh):
    '''
    data shape: B, H, W
    target shape: B, H, W
    '''
    criterion = nn.MSELoss()
    rmse = []
    for i in range(len(thsh)):
        if i != len(thsh) -1:
            x, y, z = np.where((target>=thsh[i]) & (target<thsh[i+1]))
        else:
            x, y, z = np.where(target>=thsh[i])
        torch_data = torch.from_numpy(data[x, y, z])
        torch_target = torch.from_numpy(target[x, y, z])
        rmse.append(torch.sqrt(criterion(torch_data, torch_target)).numpy())
    return rmse

class special_bar:
    def __init__(self, data):
        '''
        data shape: [N model][exps, thsh]
        '''
        self._data = data
        self._N = len(data)
        self._thsh_num = data[0].shape[1]
        
    def plot(self, tick_names, names, width=0.5):
        pace = np.ceil((self._N + 1) * width)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=200, facecolor='w')
        for i in range(self._N):
            mean = self._data[i].mean(axis=0)
            max_err = self._data[i].max(axis=0) - mean
            min_err = mean - self._data[i].min(axis=0)
            ax.bar(np.arange(self._thsh_num) * pace + i * width, 
                   mean, 
                   yerr = np.array([min_err, max_err]),
                   width=width, 
                   zorder=2,
                   ecolor='black',
                   capsize=4)
        if self._N % 2 == 0:
            ax.set_xticks(np.arange(self._thsh_num) * pace + (self._N/2 - 0.5) * width)
        elif self._N % 2 == 1:
            ax.set_xticks(np.arange(self._thsh_num) * pace + ((self._N-1)/2) * width)
        ax.set_xticklabels(tick_names)
        ax.grid(axis = 'y', zorder=0)
        ax.set_ylabel('RMSE(mm)')
        ax.set_xlabel('mm')
        ax.set_title('0-1-h RMSE of diff models (per grid)')
        ax.legend(labels=names)