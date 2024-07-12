# -*- coding: utf-8 -*-
import os, sys, glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import scipy.io as sio
import scipy.interpolate
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.extend(['/work/xc3vancechen/DLRA/training'])
from core.constants import RAIN_Q95
from core.raw_data import RawRainData, RawQpesumsData


os.environ['INPDIR'] = '/bk2/vancechen/model_output/nc_files/'
#os.environ['INPDIR'] = '/bk2/vancechen/DLRA_train/tmp_for_plot'
os.environ['TARGET'] = '/bk2/vancechen/QPESUMS/yilin' #/2022/202201/
os.environ['QPF']    = '/bk2/vancechen/QPESUMSQPF/yilin'
os.environ['RWRF']   = '/bk2/vancechen/RWRF/yilin'
os.environ['iTeen']   = '/bk2/vancechen/iTeen/yilin'
#os.environ['OUPDIR'] = '/bk2/vancechen/model_output/final_figs'
os.environ['OUPDIR'] = '/bk2/vancechen/model_output/figures'
os.environ['PLT_TOOL_DIR'] = '/wk171/vancechen/DLRA_train/pyplot_tools/static'


class plot_deepqpf:
    def __init__(self, reconstruction, oupDir, dt) -> None:
        self._recons = reconstruction
        self._oupDir = oupDir
        self._dt = dt
        
    def fig_setting(self):
        self.lon_min = 118.73 
        self.lon_max = 123.05 
        self.lat_min = 21.45
        self.lat_max = 25.63
        self.cmap = mpl.colors.ListedColormap(['#FFFFFF', '#9CFCFF', '#03C8FF', '#059BFF', '#0363FF', '#059902', 
                                               '#39FF03', '#FFFB03', '#FFC800', '#FF9500', '#FF0000', '#CC0000', 
                                               '#990000', '#960099', '#C900CC', '#FB00FF', '#FDC9FF', '#D3B9F9', 
                                               '#B685D7', '#9E5ECA', '#72339D']
                                             )
        self.bounds = [ 0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300, 400, 500, 600, 700]
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N, extend='max')  #雨量colorbar
        self.colors = ['#FFFFFF', '#9CFCFF', '#03C8FF', '#059BFF', '#0363FF', '#059902', '#39FF03', '#FFFB03', 
                       '#FFC800', '#FF9500', '#FF0000', '#CC0000', '#990000', '#960099', '#C900CC', '#FB00FF', 
                       '#FDC9FF', '#D3B9F9', '#B685D7', '#9E5ECA', '#72339D']
        
    def font_setting(self):
        font_dirs = os.path.join(os.environ['PLT_TOOL_DIR'], 'font_data')
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = 'Noto Sans TC'
        
    def set_model_region(self):
        # model axis
        latStart = 20; latEnd = 27;
        lonStart = 118; lonEnd = 123.5;
        lat = np.linspace(latStart,latEnd,561)
        lon = np.linspace(lonStart,lonEnd,441)
        self.lon_crop, self.lat_crop = np.meshgrid(lon[10:430], lat[10:550]) # refer to ccrop function
        
    def plot_rain(self):
        self.font_setting()
        self.fig_setting()
        self.set_model_region()
        img_path = self.make_dest_dirs(self._dt, self._oupDir)
        all_data = np.concatenate([self._recons, self._recons.sum(axis=0, keepdims=True)]) #[4, 560, 420]
        
        proj = ccrs.PlateCarree()
        for i in range(all_data.shape[0]):
            fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': proj})
            ax.set_facecolor('silver')
            ax.set_extent([self.lon_min, self.lon_max, self.lat_min, self.lat_max],
                          crs=ccrs.PlateCarree(),
                         )
            # ax.add_feature(cfeature.COASTLINE.with_scale('10m'), lw=1)
            rain_contour = ax.contourf(self.lon_crop, 
                                       self.lat_crop, 
                                       all_data[i], 
                                       levels=self.bounds, 
                                       colors=self.colors, 
                                       extend='max')
            cbar = plt.colorbar(rain_contour, 
                                ax=ax, 
                                orientation="horizontal",
                                shrink = 0.85,
                                pad= 0.02,  
                                ticks=self.bounds, 
                                aspect=50)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.set_xticklabels(self.bounds, weight='bold')

            # ax.coastlines(resolution='10m', linewidth =1)
            shp_path_tw0  = os.path.join(os.environ['PLT_TOOL_DIR'], 'shp_file/gadm36_TWN_0.shp')
            shp_path_tw1  = os.path.join(os.environ['PLT_TOOL_DIR'], 'shp_file/gadm36_TWN_2.shp')
            reader_tw0    = Reader(shp_path_tw0)
            reader_tw1    = Reader(shp_path_tw1)
            tw_coastline  = cfeature.ShapelyFeature(reader_tw0.geometries(), 
                                                    proj, 
                                                    edgecolor='k', 
                                                    facecolor='none')
            ax.add_feature(tw_coastline, linewidth=1)
            tw_countyline  = cfeature.ShapelyFeature(reader_tw1.geometries(), 
                                                     proj, 
                                                     edgecolor='k', 
                                                     facecolor='none')
            ax.add_feature(tw_countyline, linewidth=0.2)
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params('both', labelsize=12)

            if i == 3:
                plt.title(f'DeepQPF 三小時累積降水預報 (mm)',fontsize=10, fontweight='bold', loc='left', pad=20)
                plt.title(f'初始時間: {plot_deepqpf.dtstr_from_dt(self._dt, 2)} (UTC+8) [00 hr]\n'\
                          f'預報時間: {plot_deepqpf.dtstr_from_dt(self._dt + timedelta(minutes=60*i), 2)} '\
                          f'(UTC+8) [03 hr]\n'\
                          f'氣候天氣災害研究中心 Center for Weather Climate and Disaster Research',
                          fontsize=4, fontweight='bold', loc='right', pad=2)
                str_save_name = f'deepQPF_{plot_deepqpf.dtstr_from_dt(self._dt, 3)}'\
                                f'_f99.png'
            else:
                plt.title(f'DeepQPF 一小時累積降水預報 (mm)',fontsize=10, fontweight='bold', loc='left', pad=20)
                plt.title(f'初始時間: {plot_deepqpf.dtstr_from_dt(self._dt, 2)} (UTC+8) [00 hr]\n'\
                          f'預報時間: {plot_deepqpf.dtstr_from_dt(self._dt + timedelta(minutes=60*(i+1)), 2)} '\
                          f'(UTC+8) [{(i+1):02} hr]\n'\
                          f'氣候天氣災害研究中心 Center for Weather Climate and Disaster Research',
                          fontsize=4, fontweight='bold', loc='right', pad=2)
                str_save_name = f'deepQPF_{plot_deepqpf.dtstr_from_dt(self._dt, 3)}'\
                                f'_f{(i+1):02}'\
                                f'.png'

            fig.savefig(os.path.join(img_path, str_save_name),
                        dpi=500, 
                        bbox_inches='tight', 
                        pad_inches=0.1
                        )
            plt.close()
            
    @classmethod
    def dtstr_from_dt(cls, init:datetime, strType:int = 0):
        if strType == 0:
            return init.strftime("%j-%H%M")
        elif strType == 1:
            return init.strftime("%H%M")
        elif strType == 2:
            return init.strftime("%Y%m%d %H:%M")
        elif strType == 3:
            return init.strftime("%Y%m%d%H%M")
        elif strType == 4:
            return init.strftime("%Y%m%d%H")
        elif strType == 5:
            return init.strftime("%H")
        
    def make_dest_dirs(self, dt, parentDir):
        #fpath = os.path.join(parentDir, f"{dt.year}{dt.month:02}{dt.day:02}", f"{dt.hour:02}")
        fpath = os.path.join(parentDir, f"{dt.year}{dt.month:02}{dt.day:02}")
        if not os.path.exists(fpath):
            os.makedirs(fpath, exist_ok=True)
        return fpath

def plotFig(inp, oup, oup_dir):
    """
    Lagacy code:
        draw [-60m, -50m ... 0m, +60m, +120m, +180m] 9 figures
    """
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

    # set colorbar
    cwbRR = mpl.colors.ListedColormap(['#FFFFFF', '#9CFCFF', '#03C8FF', '#059BFF', '#0363FF',
                                       '#059902', '#39FF03', '#FFFB03', '#FFC800', '#FF9500',
                                       '#FF0000', '#CC0000', '#990000', '#960099', '#C900CC',
                                       '#FB00FF', '#FDC9FF'])
    bounds = [ 0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
    norm = mpl.colors.BoundaryNorm(bounds, cwbRR.N)
    
    # city edge
    mat = sio.loadmat('pyplot_tools/city_lonlat_region5.mat')
    citylon = mat['citylon']
    citylat = mat['citylat']
    del mat
    
    # model axis
    latStart = 20; latEnd = 27;
    lonStart = 118; lonEnd = 123.5;
    lat = np.linspace(latStart,latEnd,561)
    lon = np.linspace(lonStart,lonEnd,441)
    lon_large, lat_large = np.meshgrid(lon, lat)
    lon_crop, lat_crop = np.meshgrid(lon[10:430], lat[10:550]) # refer to ccrop function
    

    fig, ax = plt.subplots(1,9, figsize=(20, 5), dpi=200, facecolor='w')
    for y in range(6): # gt x 6, 0-1, 1-2, 2-3
        ax[y].plot(citylon, citylat,'k',linewidth=0.6)
        ax[y].axis([119, 123, 21, 26]) # whole area [119, 123, 21, 26]
        ax[y].set_aspect('equal')
        ax[y].pcolormesh(lon_large, lat_large, inp[y] * RAIN_Q95, edgecolors='none',shading='auto', 
                         norm=norm, cmap=cwbRR)
        ax[y].set_xticks([])
        ax[y].set_yticks([])
    for y in range(6, 9):
        ax[y].plot(citylon, citylat,'k',linewidth=0.6)
        ax[y].axis([119, 123, 21, 26])
        ax[y].set_aspect('equal')
        ax[y].pcolormesh(lon_crop, lat_crop, oup[y-6], edgecolors='none',shading='auto', 
                         norm=norm, cmap=cwbRR)
        ax[y].set_xticks([])
        ax[y].set_yticks([])
    plt.savefig(os.path.join(oup_dir, 'deepQPF.png'), dpi=200)

class PlotAllNetcdf(plot_deepqpf):
    """
    Read all of the netCDF files in the given directory.

    Please change the key name in "readNetcdf" function.
    """
    def __init__(self, sourceDir: str, targetDir: str, qpfDir: str, rwrfDir: str, iTDir:str):
        self._sDir = sourceDir
        self._tDir = targetDir
        self._qDir = qpfDir
        self._rwDir = rwrfDir
        self._iTDir = iTDir

    def doPlot(self, outputDir):
        fullNameList = self.listAllFiles(self._sDir)
        nnfullNameList = sorted(fullNameList)
        self.font_setting()
        self.fig_setting()
        self.set_model_region()
        sampling = 6
        dt = self.getTime(nnfullNameList[0])
        

        for nn in tqdm(range(0,len(nnfullNameList)+4,sampling)):

            #fileNameList = nnfullNameList[nn:nn+sampling]
            output_rr = np.zeros((sampling,540,420), dtype=float)
            qperr     = np.zeros((sampling,540,420), dtype=float)
            qpf       = np.zeros((sampling,540,420), dtype=float)
            rwrf      = np.zeros((sampling,540,420), dtype=float)
            iTeen     = np.zeros((sampling,540,420), dtype=float)
            
            for sr in range(0,sampling):
            
                # get time
                dst = dt + timedelta(minutes=60*sr)
                if sr == 0:
                    ini_dt = dst

                year  = dst.year
                month = dst.month
                day   = dst.day
                file_format = "deepQPF_%Y%m%d_%H%M.nc"
                fileName = os.path.join(self._sDir, f"{year}", f"{year}{month:02d}",f"{year}{month:02d}{day:02d}", 
                                        dst.strftime(file_format)
                                        )              

                if os.path.isfile(fileName):

                    # read nc & take 1st time step only
                    output_rr[sr,:,:] = RawRainData(fileName).load()['rain'][0,...]
                    # read nc & acquire the target
                    targetName    = self.get_counterpart_target(dst)
                    qperr[sr,:,:] = self.get_qperr(dst, targetName, nn, len(nnfullNameList))
                    # read nc & acquire the qpf
                    qpfName     = self.get_counterpart_qpf(dst)
                    qpf[sr,:,:] = self.get_qpf(dst, qpfName, nn, len(nnfullNameList))
                    # read nc & acquire the rwrf
                    rwrfName    = self.get_counterpart_RWRF(dst)
                    rwrf[sr,:,:]= self.get_regridded_RWRF(dst, rwrfName, nn, len(nnfullNameList))
                    # read nc & acquire the iTeen
                    iTeenName = self.get_counterpart_iTeen(dst)
                    iTeen[sr,:,:] = self.get_regridded_iTeen(dst, iTeenName, nn, len(nnfullNameList))
                    print(fileName, targetName, qpfName, rwrfName, iTeenName)
                
            qperr[qperr<0] = 0
            qpf[qpf<0]     = 0
            rwrf[rwrf<0]   = 0
            iTeen[iTeen<0] = 0
            reconstruction = np.concatenate([qperr[None,...],output_rr[None,...],qpf[None,...],rwrf[None,...],iTeen[None,...]], axis=0)  #[4,6,540,420]
            print(reconstruction.shape)      
            dt        += timedelta(minutes=60*sampling)
            
            # plot
            self._doMultiplePlot(reconstruction, ini_dt, outputDir)
         
    def doDeepQPFPlot(self, outputDir):
        fullNameList = self.listAllFiles(self._sDir)
        nnfullNameList = sorted(fullNameList)
        self.font_setting()
        self.fig_setting()
        self.set_model_region()
        sampling = 6
        dt = self.getTime(nnfullNameList[0])

        for nn in tqdm(range(0,len(nnfullNameList)+4,sampling)):

            output_rr = np.zeros((sampling,540,420), dtype=float)
            qperr     = np.zeros((sampling,540,420), dtype=float)

            for sr in range(0,sampling):

                dst = dt + timedelta(minutes=60*sr)
                if sr == 0:
                    ini_dt = dst

                year  = dst.year
                month = dst.month
                day   = dst.day
                file_format = "deepQPF_%Y%m%d_%H%M.nc"
                fileName = os.path.join(self._sDir, f"{year}", f"{year}{month:02d}",f"{year}{month:02d}{day:02d}", 
                                        dst.strftime(file_format)
                                        )

    def listAllFiles(self, file: str) -> list:
        # find all nc files
        return glob.glob(file + '/**/*.nc', recursive=True)
        # return glob.glob(self._sDir + '/**/*.nc', recursive=True)

    def readNetcdf(self, file: str) -> np.array:
        if 'qpf1h' in file:
            data = RawQpesumsData(file).load()['rain']
        #elif 'output_rr' in file:
        #    data = DeepQPFRainData(file).load()['rain']
        elif 'qperr' in file:
            data = RawRainData(file).load()['rain'] 
        data[data < 0] = 0
        return data

    def get_counterpart_target(self, dt) -> str:
        dt_t = dt + timedelta(minutes=60*1)
        dt_str = f"{dt_t.year}{dt_t.month:02}{dt_t.day:02}_{dt_t.hour:02}{dt_t.minute:02}"
        target = os.path.join(self._tDir,f"{dt_t.year}",f"{dt_t.year}{dt_t.month:02}",f"{dt_str}.nc")
        return target

    def get_qperr(self, dt: datetime, file: str, current_round, length) -> np.array:
        qperr  = np.zeros((540,420), dtype=float)
        if not(os.path.isfile(file)):
            for fn in range(length-current_round):
                file_Forward = self.get_counterpart_target(dt + timedelta(minutes=10*(fn+1)))
                if (os.path.isfile(file_Forward)):
                    break
            for bn in range(current_round):
                file_Backward = self.get_counterpart_target(dt - timedelta(minutes=10*(bn+1)))
                if (os.path.isfile(file_Backward)):
                    break
            qperr = (RawRainData(file_Forward).load()['rain'][10:550,10:430]+\
                    RawRainData(file_Backward).load()['rain'][10:550,10:430])/2.
        else: 
            qperr = RawRainData(file).load()['rain'][10:550,10:430]
        return qperr
    


    def get_counterpart_qpf(self, dt: datetime) -> list:
        # dt_t = dt + timedelta(minutes=60*1)
        dt_t = dt
        dt_str = f"{dt_t.year}{dt_t.month:02}{dt_t.day:02}_{dt_t.hour:02}{dt_t.minute:02}"
        qpf_list = sorted(glob.glob(self._qDir+"/*"))
        qpf = []
        for id, file in enumerate(qpf_list, start=1):
            tmp_qpf = os.path.join(file,f"{dt_t.year}",f"{dt_t.year}{dt_t.month:02}",f"{dt_str}_f{id}hr.nc")
            qpf.append(tmp_qpf)
        return qpf

    def get_qpf(self, dt: datetime, file: str, current_round, length) -> np.array:
        qpf  = np.zeros((540,420), dtype=float)
        if not(os.path.isfile(file)):
            qpf_dir = os.path.dirname(file)
            tail_str = os.path.basename(file).split("_")[-1]
            for fn in range(length-current_round):
                dnt = dt + timedelta(minutes=10*(fn+1))
                dnt_str = f"{dnt.year}{dnt.month:02}{dnt.day:02}_{dnt.hour:02}{dnt.minute:02}"
                file_Forward = os.path.join(qpf_dir,f"{dnt_str}_{tail_str}")
                if (os.path.isfile(file_Forward)):
                    print(file_Forward)
                    break
            for bn in range(current_round):
                dpt = dt - timedelta(minutes=10*(bn+1))
                dpt_str = f"{dpt.year}{dpt.month:02}{dpt.day:02}_{dpt.hour:02}{dpt.minute:02}"
                file_Backward = os.path.join(qpf_dir,f"{dpt_str}_{tail_str}")
                if (os.path.isfile(file_Backward)):
                    print(file_Backward)
                    break
            qpf = (RawQpesumsData(file_Forward).load()['rain'][10:550,10:430]+\
                   RawQpesumsData(file_Backward).load()['rain'][10:550,10:430])/2.
        else: 
            qpf = RawQpesumsData(file).load()['rain'][10:550,10:430]
        return qpf

    def get_counterpart_RWRF(self, dt: datetime) -> str:
        dt_str = f"{dt.year}{dt.month:02}{dt.day:02}_{dt.hour:02}{dt.minute:02}"
        rwrf = os.path.join(self._rwDir,f"{dt.year}",f"{dt.year}{dt.month:02}",f"{dt_str}RDA.nc")
        return rwrf

    def get_regridded_RWRF(self, dt: datetime, file: str, current_round, length) -> np.array:

        if not(os.path.isfile(file)):
            for fn in range(length-current_round):
                file_Forward = self.get_counterpart_RWRF(dt + timedelta(minutes=10*(fn+1)))
                if (os.path.isfile(file_Forward)):
                    break
            for bn in range(current_round):
                file_Backward = self.get_counterpart_RWRF(dt - timedelta(minutes=10*(bn+1)))
                if (os.path.isfile(file_Backward)):
                    break
            rwrf_rr = (RawRWRF(file_Forward).load()['rain']+\
                       RawRWRF(file_Backward).load()['rain'])/2.
            lat     = RawRWRF(file_Forward).load()['lat']
            lon     = RawRWRF(file_Backward).load()['lon']
        else: 
            rwrf_rr = RawRWRF(file).load()['rain']
            #time    = RawRWRF(file).load()['time']
            lat     = RawRWRF(file).load()['lat']
            lon     = RawRWRF(file).load()['lon']

        latStart = 20; latEnd = 27;
        lonStart = 118; lonEnd = 123.5;

        indlat = np.where( (lat >= latStart) * (lat <= latEnd))[0]
        indlon = np.where( (lon >= lonStart) * (lon <= lonEnd))[0]

        oldlat  = lat[indlat[0]-1:indlat[-1]+2]
        oldlon  = lon[indlon[0]-1:indlon[-1]+2]
        crop_RR = rwrf_rr[1,indlat[0]-1:indlat[-1]+2,indlon[0]-1:indlon[-1]+2]

        ilat = np.linspace(latStart,latEnd,561)
        ilon = np.linspace(lonStart,lonEnd,441)

        LON, LAT   = np.meshgrid(oldlon, oldlat)
        LONI, LATI = np.meshgrid(ilon,ilat)
        
        rgd_rwrf_rr = scipy.interpolate.griddata((LON.flatten(),LAT.flatten()),crop_RR.flatten() , (LONI,LATI), method='cubic')
        return rgd_rwrf_rr[10:550,10:430]


    def get_counterpart_iTeen(self, dt: datetime) -> str:
        dt_str = f"{dt.year}{dt.month:02}{dt.day:02}_{dt.hour:02}{dt.minute:02}"
        iTeen = os.path.join(self._iTDir,f"{dt.year}",f"{dt.year}{dt.month:02}",f"{dt_str}npm.dat")
        return iTeen

    def get_regridded_iTeen(self, dt: datetime, file: str, current_round, length) -> np.array:
        iT_rr  = np.zeros((303,263), dtype=float)
        if not(os.path.isfile(file)):
            for fn in range(length-current_round):
                file_Forward = self.get_counterpart_iTeen(dt + timedelta(minutes=10*(fn+1)))
                if (os.path.isfile(file_Forward)):
                    break
            for bn in range(current_round):
                file_Backward = self.get_counterpart_iTeen(dt - timedelta(minutes=10*(bn+1)))
                if (os.path.isfile(file_Backward)):
                    break
            iT_rr = (RawiTeen(file_Forward).readBinaryFile_npm_and_cut()['rain']+\
                     RawiTeen(file_Backward).readBinaryFile_npm_and_cut()['rain'])/2.
        else: 
            iT_rr = RawiTeen(file).readBinaryFile_npm_and_cut()['rain']

        lat      = np.arange(19.488573,27.65,0.027)
        lon      = np.arange(117.4644,125.33,0.03)

        latStart = 20; latEnd = 27;
        lonStart = 118; lonEnd = 123.5;

        indlat = np.where( (lat >= latStart) * (lat <= latEnd))[0]
        indlon = np.where( (lon >= lonStart) * (lon <= lonEnd))[0]

        oldlat  = lat[indlat[0]-1:indlat[-1]+2]
        oldlon  = lon[indlon[0]-1:indlon[-1]+2]
        crop_RR = iT_rr[indlat[0]-1:indlat[-1]+2,indlon[0]-1:indlon[-1]+2]        

        ilat = np.linspace(latStart,latEnd,561)
        ilon = np.linspace(lonStart,lonEnd,441)

        LON, LAT   = np.meshgrid(oldlon, oldlat)
        LONI, LATI = np.meshgrid(ilon,ilat)
        
        rgd_iTeen_rr = scipy.interpolate.griddata((LON.flatten(),LAT.flatten()),crop_RR.flatten() , (LONI,LATI), method='cubic')
        return rgd_iTeen_rr[10:550,10:430]


    def getTime(self, file: str) -> datetime:
        # example: data/output/qpf1h/2022/202208/20220826_0030.nc
        dt_str = os.path.basename(file)[8:-3]
        return datetime.strptime(dt_str, '%Y%m%d_%H%M')
    
    def getTimefromQpf(self, file: str) -> datetime:
        # example: data/output/qpf1h/2022/202208/20220826_0030.nc
        dt_str = os.path.basename(file).split(".")[:-5]
        return datetime.strptime(dt_str, '%Y%m%d_%H%M')

    def _doMultiplePlot(self, recons: np.array, dt: datetime, oupDir: str) -> None:

        img_path = self.make_dest_dirs(dt, oupDir)

        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(3, 6, figsize=(12,7), subplot_kw={'projection': proj})
        for y in range(len(ax)):
            for x in range(len(ax[y])):
                ax[y,x].set_facecolor('silver')
                ax[y,x].set_extent([self.lon_min, self.lon_max, self.lat_min, self.lat_max],
                                crs=ccrs.PlateCarree(),
                                )
                # ax.add_feature(cfeature.COASTLINE.with_scale('10m'), lw=1)
                rain_contour = ax[y,x].contourf(self.lon_crop, 
                                                self.lat_crop, 
                                                recons[y,x,:,:],
                                                levels=self.bounds,
                                                colors=self.colors, 
                                                extend='max')

                # ax.coastlines(resolution='10m', linewidth =1)
                shp_path_tw0  = os.path.join(os.environ['PLT_TOOL_DIR'], 'shp_file/gadm36_TWN_0.shp')
                shp_path_tw1  = os.path.join(os.environ['PLT_TOOL_DIR'], 'shp_file/gadm36_TWN_2.shp')
                reader_tw0    = Reader(shp_path_tw0)
                reader_tw1    = Reader(shp_path_tw1)
                tw_coastline  = cfeature.ShapelyFeature(reader_tw0.geometries(), 
                                                        proj, 
                                                        edgecolor='k', 
                                                        facecolor='none')
                ax[y,x].add_feature(tw_coastline, linewidth=1)
                tw_countyline  = cfeature.ShapelyFeature(reader_tw1.geometries(), 
                                                         proj, 
                                                         edgecolor='k', 
                                                         facecolor='none')
                ax[y,x].add_feature(tw_countyline, linewidth=0.2)
                ax[y,x].xaxis.set_major_formatter(LongitudeFormatter())
                ax[y,x].yaxis.set_major_formatter(LatitudeFormatter())
                ax[y,x].tick_params('both', labelsize=12)


                if y == 0:
                    #ax[y,x].set_title(f'DeepQPF 一小時累積降水預報 (mm)',fontsize=10, fontweight='bold', loc='left', pad=25)
                    ax[y,x].set_title(f'初始時間: {plot_deepqpf.dtstr_from_dt(dt + timedelta(minutes=60*x), 2)} (UTC+0) [00 hr]\n'\
                                f'截止時間: {plot_deepqpf.dtstr_from_dt(dt + timedelta(minutes=60*(x+1)), 2)} '\
                                f'(UTC+0) [{(1):02} hr]',
                                fontsize=4, fontweight='bold', loc='right', pad=2)
                
                else:
                    #ax[y,x].set_title(f'QPESUMS 一小時累積降水 (mm)',fontsize=10, fontweight='bold', loc='left', pad=25)
                    ax[y,x].set_title(f'初始時間: {plot_deepqpf.dtstr_from_dt(dt + timedelta(minutes=60*x), 2)} (UTC+0) [00 hr]\n'\
                                f'預報時間: {plot_deepqpf.dtstr_from_dt(dt + timedelta(minutes=60*(x+1)), 2)} '\
                                f'(UTC+0) [{(1):02} hr]',
                                fontsize=4, fontweight='bold', loc='right', pad=2)


        ax[0, 0].annotate('QPESUMS', (0, 0.5), xytext=(-10, 0), fontweight='bold',
                 textcoords='offset points', xycoords='axes fraction',
                ha='right', va='center', size=18, rotation=90)   
                    
        ax[1, 0].annotate('DeepQPF', (0, 0.5), xytext=(-10, 0), fontweight='bold',
                 textcoords='offset points', xycoords='axes fraction',
                ha='right', va='center', size=18, rotation=90)

        ax[2, 0].annotate('QSQPF', (0, 0.5), xytext=(-10, 0), fontweight='bold',
                 textcoords='offset points', xycoords='axes fraction',
                ha='right', va='center', size=18, rotation=90)
        """
        ax[3, 0].annotate('RWRF', (0, 0.5), xytext=(-10, 0), fontweight='bold',
                 textcoords='offset points', xycoords='axes fraction',
                ha='right', va='center', size=18, rotation=90)
        
        ax[4, 0].annotate('iTeen', (0, 0.5), xytext=(-10, 0), fontweight='bold',
                 textcoords='offset points', xycoords='axes fraction',
                ha='right', va='center', size=18, rotation=90)
        """

        cbar = plt.colorbar(rain_contour, 
                            ax=ax, 
                            orientation="horizontal",
                            shrink = 0.85,
                            pad= 0.02,  
                            ticks=self.bounds, 
                            aspect=50)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("(mm)", fontweight='bold')
        cbar.ax.set_xticklabels(self.bounds, weight='bold')
        
        str_save_name = f'deepQPF_{plot_deepqpf.dtstr_from_dt(dt, 4)}-'\
                                f'{plot_deepqpf.dtstr_from_dt(dt + timedelta(minutes=60*5), 5)}'\
                        f'_f{(1):02}'\
                        f'.png'

        fig.savefig(os.path.join(img_path, str_save_name), 
                    dpi=800, 
                    bbox_inches='tight', 
                    pad_inches=0.1
                    )
        plt.close()

if __name__ == "__main__":
    # set target subfolder, ex: qpf1h, qperr, while radar has not been implemented
    #subfolder = 'qpf1h'

    # corresponding figure name prefix, ex: qpesumsQPF, qpesumsQPE, qpesumsQPE_3hr
    #fig_prefix    = 'qpesumsQPF_1hr'

    executor = PlotAllNetcdf(os.environ['INPDIR'], os.environ['TARGET'], os.environ['QPF'], os.environ['RWRF'], os.environ['iTeen'])
    executor.doPlot(os.environ['OUPDIR'])