#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:28:46 2020

@author: handsomedong
"""
import h5py, os
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
#part3
import sys
sys.path.extend(['/bk2/handsomedong/PythonLearning/analysis/'])
from usefulFunct import classify, calRMSEbycategory, calMAPEbycategory

path = "/bk2/handsomedong/database/finalSaver/"
files = os.listdir(path)
zr_files = [path+x for x in files if x.endswith("zr.h5")]
cnn_files = [path+x for x in files if x.endswith("CNN_v2.h5")]
zr_files.sort()
cnn_files.sort()
del files, path

def kickOutlier(zr):
    Outlier_station = ['Wugu','Wufengshan','Songshan','Siapen','Quchi', 'Linkou',
                       'FBG', 'Baishawan', 'Pingdeng', 'Qidu','Shenkeng', 
                       'Sanchong', 'Datunshan']
    for i in range(len(Outlier_station)):
         ans = [x for x in zr if Outlier_station[i] in x]
         zr.remove("".join(ans))
    return zr
zr_files = kickOutlier(zr_files)

sitename=[]
for i in range(len(cnn_files)):
    #
    loc = cnn_files[i].find("_")
    sitename.append(cnn_files[i][38:loc])
del i, loc

# Load Excel
site_list = pd.read_excel('/bk2/handsomedong/database/data/NorthTW_StationSite.xlsx',
                          header=None,
                          index_col=0,
                          names=['SiteCode', 'Lon', 'Lat'])
def main2():
    terr = np.array([])
    mat = sio.loadmat('/bk2/handsomedong/database/data/terrain_TW.mat')
    terrain_Taiwan = mat['Taiwan']
    terrain_lat = np.linspace(26.0001, 20.9999, 18001)
    terrain_lon = np.linspace(118.9999, 123.0001, 14401)
    
    for i in range(len(sitename)):
        tmp_lon = site_list.loc[sitename[i]].Lon
        tmp_lat = site_list.loc[sitename[i]].Lat
        a = np.argmin(np.abs(terrain_lat-tmp_lat))    
        b = np.argmin(np.abs(terrain_lon-tmp_lon))
        terr = np.append(terr, terrain_Taiwan[a, b])
    return terr
tar_terr = main2();tar_terr[23]=241; del site_list
mountain = np.where(tar_terr>=200)[0]

def combine(cnn, zr):
    a = {'stno':_, 'yyyymmddhh':_, 'PP01':_ }
    gt_cnn1 = pd.DataFrame(columns=a)
    b = {'# stno':_, 'yyyymmddhh':_, 'PP01':_ }
    gt_zr1 = pd.DataFrame(columns=b)
    qpe_cnn1 = np.array([])
    qpe_zr1 = np.array([])
    for i in mountain:
        ### CNN
        with h5py.File(cnn[i], 'r') as hf:
            a = [key for key in hf.keys()]
            estimate_cnn = hf[a[0]][:]
        labels_cnn = pd.read_hdf(cnn[i], keys=a[1], mode='r')
        estimate_cnn[estimate_cnn<0.5]=0
        del a
        ###ZR
        with h5py.File(zr[i], 'r') as hf:
            a = [key for key in hf.keys()]
            estimate_zrr = hf[a[1]][:]
        labels_zrr = pd.read_hdf(zr[i], keys=a[0], mode='r')
        estimate_zrr[estimate_zrr<0.5]=0
        del a
        # filtering by date time
        date_cnn = labels_cnn.yyyymmddhh.values
        date_zrr = labels_zrr.yyyymmddhh.values
        tmp = np.array([])
        for j in range(len(date_cnn)):
            if date_cnn[j] not in date_zrr:
                labels_cnn = labels_cnn.drop(index=j)
                tmp = np.append(tmp, j)
        estimate_cnn = np.delete(estimate_cnn, tmp.astype(np.int16))
        del j, tmp
        tmp = np.array([])
        for j in range(len(date_zrr)):
            if date_zrr[j] not in date_cnn:
                labels_zrr = labels_zrr.drop(index=j)
                tmp = np.append(tmp, j)
        estimate_zrr = np.delete(estimate_zrr, tmp.astype(np.int16))
        del j, tmp
        
        ### concat
        gt_cnn1 = pd.concat([gt_cnn1, labels_cnn],
                             ignore_index=True)
        gt_zr1 = pd.concat([gt_zr1, labels_zrr],
                            ignore_index=True)
        qpe_cnn1 = np.append(qpe_cnn1, estimate_cnn)
        qpe_zr1 = np.append(qpe_zr1, estimate_zrr)
    return gt_cnn1, gt_zr1, qpe_cnn1, qpe_zr1
gt_cnn, gt_zr, qpe_cnn, qpe_zrr = combine(cnn_files, zr_files)

group_cnn = classify(gt_cnn.PP01.values, qpe_cnn)
group_zrr = classify(gt_cnn.PP01.values, qpe_zrr)
rmse_4class_cnn = calRMSEbycategory(gt_cnn.PP01.values, qpe_cnn)
rmse_4class_zrr = calRMSEbycategory(gt_cnn.PP01.values, qpe_zrr)
mape_4class_cnn = calMAPEbycategory(gt_cnn.PP01.values, qpe_cnn)
mape_4class_zrr = calMAPEbycategory(gt_cnn.PP01.values, qpe_zrr)
#%%
def FocusOnHeavy():
    loc = np.where(gt_cnn.PP01 >=5)
    qpe_cnn_heavy = qpe_cnn[loc]; qpe_zrr_heavy = qpe_zrr[loc]
    true_heavy = gt_cnn.PP01.values[loc]
    #5~10
#    mask_0510 = np.where((true_heavy>=5)&(true_heavy<10))
#    qpe_cnn_0510 = qpe_cnn_heavy[mask_0510]-true_heavy[mask_0510]
#    qpe_zrr_0510 = qpe_zrr_heavy[mask_0510]-true_heavy[mask_0510]
#    print(np.sqrt(np.mean(np.square(qpe_cnn_heavy[mask_0510]-true_heavy[mask_0510]))))
#    print(np.sqrt(np.mean(np.square(qpe_zrr_heavy[mask_0510]-true_heavy[mask_0510]))))
    # 10~20
    mask_1020 = np.where((true_heavy>=10)&(true_heavy<20))
    qpe_cnn_1020 = qpe_cnn_heavy[mask_1020]-true_heavy[mask_1020]
    qpe_zrr_1020 = qpe_zrr_heavy[mask_1020]-true_heavy[mask_1020]
    print(np.sqrt(np.mean(np.square(qpe_cnn_heavy[mask_1020]-true_heavy[mask_1020]))))
    print(np.sqrt(np.mean(np.square(qpe_zrr_heavy[mask_1020]-true_heavy[mask_1020]))))    
    # 20~30
    mask_2030 = np.where((true_heavy>=20)&(true_heavy<30))
    qpe_cnn_2030 = qpe_cnn_heavy[mask_2030]-true_heavy[mask_2030]
    qpe_zrr_2030 = qpe_zrr_heavy[mask_2030]-true_heavy[mask_2030]
    print(np.sqrt(np.mean(np.square(qpe_cnn_heavy[mask_2030]-true_heavy[mask_2030]))))
    print(np.sqrt(np.mean(np.square(qpe_zrr_heavy[mask_2030]-true_heavy[mask_2030]))))      
    # 30~40
    mask_3040 = np.where((true_heavy>=30)&(true_heavy<40))
    qpe_cnn_3040 = qpe_cnn_heavy[mask_3040]-true_heavy[mask_3040]
    qpe_zrr_3040 = qpe_zrr_heavy[mask_3040]-true_heavy[mask_3040]
    print(np.sqrt(np.mean(np.square(qpe_cnn_heavy[mask_3040]-true_heavy[mask_3040]))))
    print(np.sqrt(np.mean(np.square(qpe_zrr_heavy[mask_3040]-true_heavy[mask_3040]))))      
    # 40~50
    mask_4050 = np.where((true_heavy>=40)&(true_heavy<50))
    qpe_cnn_4050 = qpe_cnn_heavy[mask_4050]-true_heavy[mask_4050]
    qpe_zrr_4050 = qpe_zrr_heavy[mask_4050]-true_heavy[mask_4050]
    print(np.sqrt(np.mean(np.square(qpe_cnn_heavy[mask_4050]-true_heavy[mask_4050]))))
    print(np.sqrt(np.mean(np.square(qpe_zrr_heavy[mask_4050]-true_heavy[mask_4050]))))     
    # 50~60
    mask_5060 = np.where((true_heavy>=50)&(true_heavy<60))
    qpe_cnn_5060 = qpe_cnn_heavy[mask_5060]-true_heavy[mask_5060]
    qpe_zrr_5060 = qpe_zrr_heavy[mask_5060]-true_heavy[mask_5060]
    print(np.sqrt(np.mean(np.square(qpe_cnn_heavy[mask_5060]-true_heavy[mask_5060]))))
    print(np.sqrt(np.mean(np.square(qpe_zrr_heavy[mask_5060]-true_heavy[mask_5060]))))  
    
#    group1 = [#qpe_cnn_0510.tolist(),
#              qpe_cnn_1020.tolist(), 
#              qpe_cnn_2030.tolist(), 
#              qpe_cnn_3040.tolist(), 
#              qpe_cnn_4050.tolist(),
#              qpe_cnn_5060.tolist()]
#    group2 = [#qpe_zrr_0510.tolist(),
#              qpe_zrr_1020.tolist(), 
#              qpe_zrr_2030.tolist(), 
#              qpe_zrr_3040.tolist(), 
#              qpe_zrr_4050.tolist(),
#              qpe_zrr_5060.tolist()]
    group1 = [qpe_cnn_1020.tolist(), 
              qpe_cnn_2030.tolist(), 
              qpe_cnn_3040.tolist(), 
              qpe_cnn_4050.tolist(),
              qpe_cnn_5060.tolist()]
    group2 = [qpe_zrr_1020.tolist(), 
              qpe_zrr_2030.tolist(), 
              qpe_zrr_3040.tolist(), 
              qpe_zrr_4050.tolist(),
              qpe_zrr_5060.tolist()]
    
    font = {'family'     : 'DejaVu Sans Mono',
            'weight'     : 'bold',
            'size'       : 14
            }
    axes = {'titlesize'  : 18,
            'titleweight': 'heavy',
            'labelsize'  : 16,
            'labelweight': 'bold'
            }
    mpl.rc('font', **font)  # pass in the font dict as kwargs
    mpl.rc('axes', **axes)
    
    ticks_name = ['10~20 mm', '20~30 mm', '30~40 mm', '40~50 mm', '50~60 mm']
    fig, ax = plt.subplots(1, 1, num='result', figsize=(9, 6), dpi=100, facecolor='w')
    fig.tight_layout()
    bplot1 = ax.boxplot(group1,
                           positions=np.array(range(len(group1)))*2.0-0.25-0.01,
                           sym='', 
                           whis=1.5,
                           widths=0.5,
                           patch_artist=True)
    bplot2 = ax.boxplot(group2,
                           positions=np.array(range(len(group1)))*2.0+0.25+0.01,
                           sym='', 
                           whis=1.5, 
                           widths=0.5,
                           patch_artist=True)
    colors1 = ['red', 'red', 'red', 'red', 'red']
    colors2 = ['blue', 'blue', 'blue', 'blue','blue']
    for patch, color in zip(bplot1['boxes'], colors1):
        patch.set_facecolor(color)# fill some different colors
        patch.set_alpha(0.9)
    for patch, color in zip(bplot2['boxes'], colors2):
        patch.set_facecolor(color)# fill some different colors
        patch.set_alpha(0.9)
    plt.setp(bplot1['medians'], color='yellow')
    plt.setp(bplot2['medians'], color='yellow')
    ax.set_xlim(-1.5,9.5)
    ax.set_xticks(range(0, len(ticks_name)*2, 2))
    ax.set_xticklabels(ticks_name)
    ax.set_ylabel('Bias (mm)')
    ax.set_ylim([-60, 30])
    #ax.set_title('Biases of 13 Mountain stations',pad=40)
    ax.legend(handles=[bplot1["boxes"][0], bplot2["boxes"][0]], 
              labels=['CNN model', 'Z-R relation'], 
              loc='lower left',
              markerfirst=False,
              fontsize='15')
    ax.text(-1.4, 31, '# '+str(len(group1[0]))+":"+str(len(group1[1]))+":"+
            str(len(group1[2]))+":"+str(len(group1[3]))+':'+str(len(group1[4])), fontsize=14, )
    # Auxiliary line
    pl = ax.plot(np.arange(-2, 17), np.full([len(range(-2,17))], 0), '-.', c='blue')
    ax.grid(axis='y')
    return 
FocusOnHeavy()
