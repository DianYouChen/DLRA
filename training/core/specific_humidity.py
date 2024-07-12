# This program is about dealing with the specific humidity (q).
# q = 0.622e/p
# e = es*(RH/100)
# es = 6.11*exp(-Lv/Rv(1/T-1/T0))

# import necessary modules
import os
import numpy as np
import netCDF4 as nc
from tqdm import tqdm
import datetime

from core.constants import ERA_DIR, ERA_QV_DIR
# turn off warnings
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()

class readERAFile:
    var_list = ['level', 't', 'r', 'u', 'v']
    def __init__(self, fpath:str, dim:tuple = (29, 23)):
        self._dim = dim
        self._fpath = fpath

    def load(self):
        data_rep = []
        f = nc.Dataset(self._fpath)
        for var in self.var_list:
            tmp = f[var][:]
            tmp = self.invalid_value(tmp)
            data_rep.append(tmp)
        assert np.shape(data_rep[1][0,0]) == self._dim, f'{self._fpath} has a wrong shape.'
        return data_rep
    
    def invalid_value(self, data_ma:np.ma):
        data_ma[np.where(data_ma.mask != 0)] = data_ma.fill_value
        return data_ma.data
    
def specific_humidity(p:np.array, t:np.array, rh:np.array):
    es0 = 6.11 # hpa
    Lv = 2.5e6 # J/kg
    Rv = 461 # J/K*kg
    T0 = 273.15 # K
    assert np.min(t) >= 0, 'Kelvin Temperature.'
    es = es0 * np.exp(-1 * Lv / Rv * (1 / t - 1 / T0))
    e = es * rh / 100
    q = 0.622 * e / p
    return q

class save_qv:
    def __init__(self, orig_path:str, dest_path:str, data):
        self._opath = orig_path
        self._dpath = dest_path
        self._data = data
        self._dname = self.check_dir(self._opath, self._dpath)
        self.saving(self._data, self._dname)
    
    def check_dir(self, opath, dpath):
        # opath = /bk2/handsomedong/DLRA_database/era5/2015/201501/era5_20150103.nc
        basename = os.path.basename(opath).split('.')[0]
        y = basename[-8:-4]
        m = basename[-4:-2]
        d = basename[-2:]
        d_dpath = os.path.join(dpath, y, y+m)
        if not os.path.exists(os.path.join(dpath, y)):
            os.mkdir(os.path.join(dpath, y))
        if not os.path.exists(d_dpath):
            os.mkdir(d_dpath)
        return os.path.join(d_dpath, basename + '.npy')
    
    def saving(self, data, fpath):
        np.save(fpath, data)
        
class lookForNearestTime(readERAFile):
    def __init__(self, dt:datetime.datetime, fp:str = ERA_DIR, fp2:str = ERA_QV_DIR):
        # fp is where the era5 data locates
        self._time = dt
        self._fp = fp
        self._fp2 = fp2
        self._full_fp = self.full_path(self._time, self._fp) 
        self._full_fp2 = self.full_path(self._time, self._fp2) 
        super().__init__(self._full_fp)
     
    def full_path(self, dt, fp):
        if 'qv' not in fp:
            era5_path = os.path.join(fp, 
                                     f'{dt.year}', 
                                     f'{dt.year:4d}{dt.month:02d}',
                                     f'era5_{dt.year:4d}{dt.month:02d}{dt.day:02d}.nc'
                                    )
        else:
            era5_path = os.path.join(fp, 
                                     f'{dt.year}', 
                                     f'{dt.year:4d}{dt.month:02d}',
                                     f'era5_{dt.year:4d}{dt.month:02d}{dt.day:02d}.npy'
                                    )
        return era5_path
    
    def collectingData(self):
        data_base = []
        _, t, _, u, v = self.load() # [p, t, rh, u, v][24, 20, 29, 23]=[time, press, x, y]
        q = np.load(self._full_fp2) # [24, 20, 29, 23]
        
        idx = self.nearestTime()
        data_base.extend(map(lambda x: x[idx][None, ...], [t, q, u, v]))
        data_base = np.concatenate(data_base, axis=0)
        return data_base # [4, 20, 29, 23]
        
        
    def nearestTime(self):
        return self._time.hour
        
if __name__ == '__main__':
    era5_shape = (29, 23)
    era_dir = '/bk2/handsomedong/DLRA_database/era5'
    dest_dir = '/bk2/handsomedong/DLRA_database/era5_qv'
    
    # original data
    era_yyyy = os.listdir(era_dir)
    era_yyyy.sort()
    era_yyyymm = []
    for yyyy in era_yyyy:
        yyyymm = os.listdir(os.path.join(era_dir, yyyy))
        yyyymm.sort()
        era_yyyymm.extend([os.path.join(era_dir, yyyy, x) for x in yyyymm]) # 6yr * 12 mth = 72
    for yyyymm in tqdm(era_yyyymm, ncols=60):
        files = os.listdir(yyyymm)
        for file in files:
            full_path = os.path.join(yyyymm, file)
            d = readFile(full_path, era5_shape)
            data = d.load() # [p, t, rh, u, v][24, 20, 29, 23]=[time, press, x, y]

            qv = np.zeros_like(data[1])
            for i in range(len(data[0])):
                qv[:, i, :, :] = specific_humidity(data[0][i],
                                                   data[1][:, i, :, :],
                                                   data[2][:, i, :, :],
                                                  )
            _ = save_qv(full_path, dest_dir, qv)