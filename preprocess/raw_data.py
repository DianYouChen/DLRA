import os
import struct
from datetime import datetime
import numpy as np

from netCDF4 import Dataset


class RawRadarData:
    def __init__(self, fpath: str):
        """
        Example fpath: 'RADARCV_2018/201809/20180908_1300.nc'
        """
        self._fpath = fpath

    def datetime(self):
        assert self._fpath[-3:] == '.nc', self._fpath
        dt_str = os.path.basename(self._fpath)[:-3]
        return datetime.strptime(dt_str, '%Y%m%d_%H%M')

    def load(self):
        assert os.path.exists(self._fpath), f'{self._fpath} does not exist!'
        data = Dataset(self._fpath, 'r')
        CV = data.variables['cv'][:]
        CV[np.where(CV.mask!=0)] = -99
        lon = data.variables['lon'][:]
        lat = data.variables['lat'][:]
        assert np.shape(CV) == (561, 441), f'{self._fpath} has a wrong shape'
        return {'radar': np.array(CV), 'lat': np.array(lat), 'lon': np.array(lon)}


class RawRainData:
    def __init__(self, fpath: str):
        """
        Example fpath: 'QPESUMS_rain_2018/201809/20180908_1300.nc'
        """
        self._fpath = fpath

    def datetime(self):
        assert self._fpath[-3:] == '.nc', self._fpath
        dt_str = os.path.basename(self._fpath)[:-3]
        return datetime.strptime(dt_str, '%Y%m%d_%H%M')

    def load(self):
        assert os.path.exists(self._fpath), f'{self._fpath} does not exist!'
        data = Dataset(self._fpath, 'r')
        RR = data.variables['qperr'][:] # maskarray
        RR[np.where(RR.mask!=0)] = -99
        lon = data.variables['lon'][:]
        lat = data.variables['lat'][:]
        assert np.shape(RR) == (561, 441), f'{self._fpath} has a wrong shape'
        return {'rain': np.array(RR), 'lat': np.array(lat), 'lon': np.array(lon)}


class RawQpesumsData:
    def __init__(self, fpath: str):
        """
        Example fpath: '20180114_0650_f1hr.nc'
        """
        self._fpath = fpath

    def datetime(self):
        assert self._fpath[-3:] == '.nc', self._fpath
        dt_str = os.path.basename(self._fpath)[:-3]
        dt_str = '_'.join(dt_str.split('_')[:2])
        return datetime.strptime(dt_str, '%Y%m%d_%H%M')

    def load(self):
        assert os.path.exists(self._fpath), f'{self._fpath} does not exist!'
        data = Dataset(self._fpath, 'r')
        RR = data.variables['qpfrr'][:]
        return {'rain': np.array(RR), 'dt': self.datetime()}
