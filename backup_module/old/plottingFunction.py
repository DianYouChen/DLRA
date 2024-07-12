# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from tqdm import tqdm

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

def plotFig(dt, inp, back): #在funct中的變數執行完就會被清除
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
    
    assert len(dt) == 8, f'only 8 frames displayed.'
    fig, ax = plt.subplots(4,8, figsize=(15, 7.5), dpi=200, facecolor='w')
    for time_step in range(len(dt)):
        for y in range(4): # gt, 0-1, 1-2, 2-3
            back_off = (lambda x: int(back / 2 * (x - 1)) if x > 0 else 0)(y)
            ax[y,time_step].plot(citylon,citylat,'k',linewidth=0.6)
            ax[y,time_step].axis([120.6875, 122.1875, 24.0625, 25.5625])# whole area [119, 123, 21, 26]
            ax[y,time_step].set_aspect('equal')
            ax[y,time_step].pcolormesh(lon, lat, inp[time_step + back - back_off, y], edgecolors='none',
                                    shading='auto', norm=norm, cmap=cwbRR)
            ax[y,time_step].set_xticks([])
            ax[y,time_step].set_yticks([])

def HSS_line_plot(names, *data: list, numbers, target_len=3) -> (None):
    # basic setting
    data_num = len(data)
    x_tick = np.array([1,3,5,10,15,20,30,40], dtype=np.int16)
    colors = ['C'+str(i) for i in range(data_num)]
    linestyles = ['-', '--', ':'] # target_length
    
    # set figure and axes
    fig, ax = plt.subplots(1,1, figsize=(10, 7.5), dpi=300, facecolor='w')
    ax2 = ax.twinx()
    ax.set_zorder(1)  # default zorder is 0 for ax and ax2，數字大的在上面
    ax.patch.set_visible(False)  # prevents ax from hiding ax2 # ax.set_frame_on(False) for new version
    
    # another legend
    for t in range(target_len):
        none = ax2.plot(-51, 0, ls=linestyles[t], color='k',lw=4)
    ax2.legend(['0-1 hr', '1-2 hr', '2-3 hr'],loc='best',fontsize=15,handlelength=2.5)
    
    # main plot code
    for i in range(data_num): # how many different data
        for t in range(target_len): # target length
            pt = ax.plot(range(len(x_tick)), data[i][t], color=colors[i], 
                          linestyle=linestyles[t], marker='o', lw=4, label=names[i], 
                          markersize=10, zorder=1,
                        ) # zorder只能用在同一個ax中
    ba = ax2.bar(np.arange(len(x_tick))+0.5, numbers, color='gray', alpha=0.5, width=0.5)
    ax.grid(axis='y', ls='--')
    ax.set_title('CSI scores')
    ax.set_xlabel('threshold (mm)')
    ax.set_axisbelow(True)
    ax.set_xlim(left=-0.5, right=len(x_tick))
    ax.set_ylim(bottom=0)
    ax.set_xticklabels(np.append(np.array([0]), x_tick))
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylabel('number of cases')

    # set legend order
    handles, labels = ax.get_legend_handles_labels()
    order = list(np.arange(0, 3*data_num, 3))
    ax.legend([handles[idx] for idx in order],
              [labels[idx] for idx in order],
              ncol=3,
              frameon=True, #邊框
              fancybox=False, #是否圓邊
              edgecolor='black',
              bbox_to_anchor=(0, -0.3, 1, 0.2), # [x0, y0, width, height]
              loc='best', # 這邊的upper left是指(x0,y0)座標對應到legend的哪個角落
              mode='expand', # expand才會讓圖例展開
              handlelength=3,
              #prop={'weight':'heavy'}
             )
'''
#     ax2.set_ylim([0,0.02])
#     vals = ax2.get_yticks()
#     ax2.set_yticklabels(['{:,.2%}'.format(a) for a in vals])
#     plt.tight_layout() # 使子圖合適地跟圖形匹配
#     plt.savefig(ROOT_DIR+'/HSS_JFMOND.png', 
#                 dpi=300, 
#                 format='png', 
#                 bbox_extra_artists=(lg,), 
#                 bbox_inches='tight')
'''

def PFD_2D(preds: list, 
           label: np.array, 
           thresholds: int = [1,3,5,10,15,20,30,40], 
           names: list = []
           ) -> (None):
    '''
    This functin is for single label plotting.
    Preds is a list containing 3d numpy arrays.
    * preds size = [N models][time, X, Y]
    * label size = [time, X, Y]
    '''
    N = len(preds)
    colors = ['C' + str(i) for i in range(N)]
    _, ax = basic_background()
    for id, pred in enumerate(tqdm(preds, ncols=60)):
        assert pred.shape == label.shape, 'Size inconsistency.'
        for threshold in thresholds:
            # numbers
            hits = np.sum(pred[label>=threshold]>=threshold)
            misses = np.sum(pred[label>=threshold]<threshold)
            false_alarms = np.sum(pred[label<threshold]>=threshold)
            correct_negatives = np.sum(pred[label<threshold]<threshold)
            #assert hits+misses+false_alarms+correct_negatives == np.size(pred)
            success_ratio = 1 - (false_alarms / (hits + false_alarms))
            probability_detection = hits / (hits + misses)
            ax.text(success_ratio, probability_detection, str(threshold),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color=colors[id],
                    fontsize=18,
                    fontweight='normal'
                   )
        # fake line for legend
        ax.plot([],[],color=colors[id],label=' ',lw=3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=names, ncol=3,
              frameon=True, #邊框
              fancybox=False, #是否圓邊
              edgecolor='black',
              bbox_to_anchor=(0, -0.3, 1, 0.2), # [x0, y0, width, height]
              mode='expand', # expand才會讓圖例展開
              loc='best', # 這邊的upper left是指(x0,y0)座標對應到legend的哪個角落
              handlelength=1, fontsize=12
             )

def maxpool(input1, filter_size, filter_stride):
    padding_size = int((filter_size-1)/2)
    tmp = np.pad(array=input1, 
                 pad_width=((padding_size,padding_size),(padding_size,padding_size)), 
                 mode='constant', 
                 constant_values=0)
    length = np.shape(tmp)[0]
    width = np.shape(tmp)[1]
    repository = np.zeros([int((length-filter_size+1)/filter_stride),
                           int((width-filter_size+1)/filter_stride)])
    for m in range(np.size(repository, 0)):
        for n in range(np.size(repository, 1)):
            repository[m, n] = np.max(input1[m*filter_stride:m*filter_stride+filter_size, 
                                             n*filter_stride:n*filter_stride+filter_size])
    return repository

def basic_background() -> (axes):
    ### Samples
    sample_bias = np.array([0.3,0.5,0.8,1,1.3,1.5,2.,3.,5.,10.])
    sample_suc = np.arange(0,1.1,0.1)
    sample_pod = np.full([len(sample_bias), len(sample_suc)], 0.)
    for j in range(len(sample_bias)):
        sample_pod[j, :] = sample_bias[j]*sample_suc
    sample_CSI=np.arange(0.1,1,0.1)
    sample_suc2 = np.arange(0.1,1.01,0.01)
    sample_pod2 = np.full([len(sample_CSI), len(sample_suc2)], 0.)
    for k in range(len(sample_CSI)):
        sample_pod2[k, :] = 1/((1/sample_CSI[k])-(1/sample_suc2)+1)
    ### Baground
    fig, ax = plt.subplots(1, 1, num='result', figsize=(7, 5.6), dpi=200, facecolor='w')
    fig.tight_layout()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel("Success Ratio")
    ax.set_ylabel("Probability of Detection") 
    ax.grid(visible=True, axis="y", alpha=0.4)
    ax.set_title('Performance Diagram', pad=18.) 
    ### Auxiliary line
    for k in range(len(sample_CSI)):
        maxloc = np.argmax(sample_pod2[k,:])
        ax.plot(sample_suc2[maxloc:], sample_pod2[k,maxloc:], '-', c='green', lw=2, alpha=0.8) 
    for j in range(len(sample_bias)):
            ax.plot(sample_suc, sample_pod[j,:], '--', c='grey', lw=2, alpha=0.8)                
    for n in range(4):
        ax.text(1, sample_bias[n], '%.1f'%sample_bias[n],color='grey',alpha=.8)
    for m in range(4,10):
        ax.text(1/sample_bias[m], 1.005, "%.1f"%sample_bias[m],
                horizontalalignment='center',
                color='grey',
                alpha=.8)    
    ### Contents
    for k in range(len(sample_CSI)):
        ax.text(1,sample_CSI[k],'%.1f'%sample_CSI[k],
                horizontalalignment='right',
                verticalalignment='center',
                color='green',alpha=0.8) 
    return fig, ax 

def boxPlot(groups: list,  
            groupNames: list = [],
            tickNames: list = []
           ) -> (None):
    '''
    Groups is a list contain several lists from different models.
    Each list(model) has lists for  numpy ERROR array.
    * group size = [N models][M thresholds][array]
    '''
    ### settings
    boxWidth = 0.5
    boxPad = 0.01
    groupNum = len(groups)
    print(groupNum)
    legend_handles = [] # only for plot legend
    ### plot
    fig, ax = plt.subplots(1, 1, num='result', figsize=(9, 6), dpi=200, facecolor='w')
    fig.tight_layout()
    if groupNum % 2 == 1:
        baseline = np.floor(groupNum / 2).astype(np.int32)
        interval = np.ceil(boxWidth + (groupNum - 1) * (boxWidth + boxPad)).astype(np.int32)
        for id, group in enumerate(tqdm(groups, ncols = 60)):
            shift1 = np.sign(id - baseline) * (boxWidth / 2)
            shift2 = np.sign(id - baseline) * (np.abs(id - baseline) - 1) * (boxWidth + boxPad)
            shift3 = np.sign(id - baseline) * (boxWidth / 2 + boxPad)
            shift_all = shift1 + shift2 + shift3
            legend_handles.append(ax.boxplot(group,
                                    positions = np.arange(len(group)) * interval + shift_all,
                                    sym = '', 
                                    whis = [5, 95],
                                    widths = boxWidth,
                                    boxprops = dict(facecolor = 'C' + str(id), alpha = 0.6),
                                    medianprops = dict(color='firebrick'),
                                    patch_artist = True)
            )
    else:
        baseline = (groupNum - 1) / 2
        interval = np.ceil(groupNum * (boxWidth + boxPad)).astype(np.int32)
        # raise RuntimeError('Not developed yet.')
        for id, group in enumerate(tqdm(groups, ncols = 60)):
            reset_idx = np.sign(id - baseline) * np.ceil(np.abs(id - baseline)) # center = 0
            shift2 = np.sign(reset_idx) * (np.abs(reset_idx) - 1) * (boxWidth + boxPad)
            shift3 = np.sign(reset_idx) * (boxWidth / 2 + boxPad)
            shift_all = shift2 + shift3
            legend_handles.append(ax.boxplot(group,
                                    positions = np.arange(len(group)) * interval + shift_all,
                                    sym = '', 
                                    whis = [5, 95],
                                    widths = boxWidth,
                                    boxprops = dict(facecolor = 'C' + str(id), alpha = 0.6),
                                    medianprops = dict(color='firebrick'),
                                    patch_artist = True)
            )
    ### background setting
    ax.set_xlim(np.floor(0 - shift_all - boxWidth/2), # (midLine - shift - halfBoxWidth)
                np.ceil((len(tickNames) -1) * interval + shift_all + boxWidth/2)
                )
    ax.set_xticks(range(0, len(tickNames) * interval, interval))
    ax.set_xticklabels(tickNames, fontsize = 12)
    ax.tick_params(axis='y', labelsize = 12) # ytick label
    ax.set_ylabel('Bias (mm)', fontsize = 14)
    ax.set_xlabel('mm', fontsize = 14)
    # ax.set_ylim(-45, 10) # 0-1 hr
    # ax.set_ylim(-55, 10) # 1-2 hr
    if '130' in tickNames[-1]:
        ax.set_ylim(bottom=-150) # 2-3 hr
    ax.set_title('Boxplot of Biases')
    ax.grid(axis='y', ls = '--')
    ax.legend(handles = [legend_handle['boxes'][0] for legend_handle in legend_handles], 
              labels = groupNames, 
              loc = 'lower left',
              markerfirst = False,
              edgecolor = 'black',
              fontsize = 12
              )
    # ax.text(-1.4, 31, '# '+str(len(group1[0]))+":"+str(len(group1[1]))+":"+
    #         str(len(group1[2]))+":"+str(len(group1[3]))+':'+str(len(group1[4])), fontsize=14, )
    # Auxiliary line
    vals = ax.get_ylim()
    ax.set_ylim(top = vals[1])
    ax.fill_between(range(0-interval, len(tickNames) * interval), 
                    0, vals[1], facecolor='C9',
                    alpha=0.3)
