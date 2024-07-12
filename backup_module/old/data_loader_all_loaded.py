from datetime import datetime, timedelta
from unittest import result
import numpy as np
import torch.utils.data as data
import os, sys
from netCDF4 import Dataset
path = os.getcwd() # return current work dir = '/wk171/.../evaluate or training'
path = path.split('/')[:-1]
# add preprocess into path so that dataloader can read slope
sys.path.append('/'.join(path))

#for radar crop
#os.environ['ROOT_DATA_DIR'] = '/bk2/vancechen/QPESUMS/raw/'

from core.radar_data_aggregator import CompressedAggregatedRadarData
from core.compressed_rain_data import CompressedRainData
from core.constants import (RADAR_Q95, RAIN_Q95, TIME_GRANULARITY_MIN,
                            TERRAIN_FILE, ERA_DIR, NX, NY)
from core.dataset import load_data
from core.enum import DataType
from preprocess.utils_data_collect.terrain_slope import load_shp, mapping

def Altitude_data(keys):
        alt_loader = load_shp(TERRAIN_FILE)
        container = []
        for key in keys:
            target_map, latList, lonList = alt_loader.getMap(key)
            new_map, new_lat, new_lon = mapping(latList, lonList, target_map, (561, 441))
            container.append(new_map[None])
        return np.concatenate(container, axis=0) # size [3, 120, 120]
    
def cal_slope(altitude):
    # For dim=1, idx 0 is south; idx -1 is north
    ns_shift = np.zeros([altitude.shape[0], altitude.shape[1]])
    ew_shift = np.zeros([altitude.shape[0], altitude.shape[1]])
    ns_shift[:-1] = altitude[1:]
    ew_shift[:, :-1] = altitude[:, 1:]
    ns_slope = -(altitude - ns_shift) # north - south
    ew_slope = -(altitude - ew_shift) # east - west
    
    ns_slope = blurness(ns_slope, k_size=5)
    ew_slope = blurness(ew_slope, k_size=5)
    return ew_slope, ns_slope
    
def blurness(data, k_size=3):
    # Moving Average
    # data shape = [H, W]
    pd = k_size//2
    tmp = np.pad(data, ((pd,pd), (pd,pd)), 'constant')
    tmpp = np.copy(tmp)
    for i in range(pd, tmp.shape[0]-pd):
        for j in range(pd, tmp.shape[1]-pd):
            tmpp[i, j] = tmp[i-pd:i+pd+1, j-pd:j+pd+1].mean()
    return tmpp[pd:-pd, pd:-pd]

def save_nc(fname, data, nlat, nlon, vname):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

    latStart = 20; latEnd = 27;
    lonStart = 118; lonEnd = 123.5;
    lat = np.linspace(latStart,latEnd,nlat)
    lon = np.linspace(lonStart,lonEnd,nlon)

    f = Dataset(fname, 'w', format = 'NETCDF4')
    f.createDimension('lat', len(lat))   
    f.createDimension('lon', len(lon))
    f.createVariable(vname, np.float32, ('lat', 'lon')) 
    f.createVariable('lat', np.float32, ('lat'))  
    f.createVariable('lon', np.float32, ('lon'))
    f.variables['lat'][:] = np.array(lat)
    f.variables['lon'][:] = np.array(lon)
    f.variables[vname][:] = ma.masked_array(data, mask=None)
    f.close()

class CropCenterNumpy:
    def __init__(self, crop_shape):
        self._cropx, self._cropy = crop_shape

    def __call__(self, img):
        #x, y = img.shape[-2:]
        #startx = x // 2 - self._cropx // 2
        #starty = y // 2 - self._cropy // 2
        #crop_image = img[..., startx:startx + self._cropx, starty:starty + self._cropy] #[21, 540, 420]
        #return crop_image
        #return img[...,325:445, 215:335] #[21,120,120]
        return img

    def crop_domain(self, domain):
        x, y = domain.shape[-2:]
        startx = x // 2 - self._cropx // 2
        starty = y // 2 - self._cropy // 2
        crop_domain = domain[..., startx:startx + self._cropx, starty:starty + self._cropy] #[..., 540, 420]
        return crop_domain


def moving_average(a, n=6):
    output = np.zeros_like(a)
    for i in range(a.shape[0]):
        output[i:i + n] += a[i:i + 1]

    output[n:] = output[n:] / n
    output[:n] = output[:n] / np.arange(1, n + 1).reshape(-1, 1, 1)
    return output


class DataLoaderAllLoaded(data.Dataset):
    def __init__(self,
                 start: datetime,
                 end: datetime,
                 input_len,
                 target_len,
                 target_offset=0,
                 target_avg_length=6,
                 threshold=0.5,
                 data_type=DataType.NoneAtAll,
                 residual=False,
                 random_std=0,
                 is_train=False,
                 is_test=False,
                 is_validation=False,
                 hourly_data=False,
                 hetero_data=False,
                 img_size=None,
                 sampling_rate=None,
                 workers=8):
        super().__init__()
        self._s = start
        self._e = end
        self._ilen = input_len
        self._tlen = target_len
        self._toffset = target_offset
        self._tavg_len = target_avg_length
        self._sampling_rate = sampling_rate
        self._train = is_train
        self._random_std = random_std
        self._dtype = data_type
        self._img_size = img_size
        if self._sampling_rate is None:
            self._sampling_rate = self._ilen

        # Predict difference in average rain rate. Difference is taken from last hour's rain rate.
        self._residual = residual
        self._threshold = threshold
        self._index_map = []
        # There is an issue in python 3.6.10 with multiprocessing. workers should therefore be set to 0.
        self._dataset = load_data(
            self._s,
            self._e,
            is_validation=is_validation,
            is_test=is_test,
            is_train=is_train,
            workers=0,
        )
        self._ccrop = CropCenterNumpy(self._img_size)
        self._time = list(self._dataset['radar'].keys())
        self._raw_altitude = Altitude_data(['高程', '坡度', '坡向'])
        self._slope_x, self._slope_y = cal_slope(self._raw_altitude[0])
        # whether to also give last 5 hours hour averaged rain rate
        self._hourly_data = hourly_data
        self._hetero_data = hetero_data
        self._set_index_map()

        DataType.print(self._dtype, prefix=self.__class__.__name__)
        print(f'[{self.__class__.__name__}] {self._s}<->{self._e} ILen:{self._ilen} TLen:{self._tlen} '
              f'Toff:{self._toffset} TAvgLen:{self._tavg_len} Residual:{int(self._residual)} Hrly:{int(hourly_data)} '
              f'Sampl:{self._sampling_rate} RandStd:{self._random_std} Th:{self._threshold}')

    def _set_index_map(self):
        raw_idx = 0
        skip_counter = 0
        target_offset = self._ilen + self._toffset + self._tavg_len * self._tlen
        while raw_idx < len(self._time):
            if raw_idx + target_offset >= len(self._time):
                break

            if self._time[raw_idx + target_offset] - self._time[raw_idx] != timedelta(seconds=TIME_GRANULARITY_MIN *
                                                                                     target_offset * 60):
                skip_counter += 1
            else:
                self._index_map.append(raw_idx)

            raw_idx += 1
            
        print(f'[{self.__class__.__name__}] Size:{len(self._index_map)} Skipped:{skip_counter}')
        
    ''' This part aims to balance the 0/1 sampling bias, but it turns out to failure
        if self._train:
            total_number = len(self._index_map)
            map_idx = 0
            container = {'yes': [], 'no':[]}
            while map_idx < total_number:
                if self._set_deeper_index_map(map_idx, target_offset):
                    container['yes'].append(self._index_map[map_idx])
                else:
                    container['no'].append(self._index_map[map_idx])
                map_idx += 1
            print(len(container['no']), len(container['yes']))
            if len(container['no']) >= len(container['yes']):
                random_idx = list(np.random.choice(container['no'], len(container['yes']), replace=False))
                self._index_map =  random_idx + container['yes']
                skip_counter += (len(container['no']) - len(container['yes']))
                assert len(self._index_map) + len(skip_counter) == len(self._time)
            else:
                raise RuntimeError("Strange~ no rain cases are less than rainy cases.")
        
    def _set_deeper_index_map(self, id, t_off):
        target_range = range(self._index_map[id] + self._ilen, self._index_map[id] + t_off)
        # check if there is rain in target map (0-1, 1-2, 2-3)
        for i in target_range:
            key = self._time[i]
            raw_rainData = self._dataset['rain'][key] # shape [3, n]
            if len(raw_rainData) == 0:
                continue
            elif len(raw_rainData) == 3 and np.sum(raw_rainData[-1] >=1) >= 12:
                return True
        return False
    '''
    
    def six_multiplication(self, input_data, factor=2000):
        # add a dim of 6 at the very beginning
        output_map = [input_data for _ in range(self._ilen)]
        output_map = np.stack(output_map, axis=0)/factor 
        return self._ccrop.crop_domain(output_map) # [..., 540, 420]

    def _get_raw_radar_data(self, index):
        key = self._time[index]
        raw_data = self._dataset['radar'][key]
        # 2D radar & resize
        radar = CompressedAggregatedRadarData.load_from_raw(raw_data)
        radar[radar < 0] = 0    
        return self._ccrop.crop_domain(radar) # [..., 540, 420]
        # 3D radar for 21lv
        #return CompressedRadarData.load_from_raw(raw_data).transpose(2,0,1) # [NZ, NX, NY]
        # 3D radar for 5lv
        #return CompressedRadarData.load_from_raw(raw_data).transpose(2,0,1)[0:10:2] # [NZ, NX, NY]

    def _input_end(self, index):
        return index + self._ilen

    def _input_range(self, index):
        return range(index, self._input_end(index))

    def _get_radar_data(self, index):
        return self._get_radar_data_from_range(self._input_range(index))

    def _get_radar_data_from_range(self, index_range):
        radar = [self._ccrop(self._get_raw_radar_data(idx))[None, ...] for idx in index_range]
        radar = np.concatenate(radar, axis=0) # [6, 120, 120]
        radar[radar < 0] = 0
        radar = radar / RADAR_Q95
        return radar

    def _get_raw_rain_data(self, index):
        key = self._time[index]
        raw_data = self._dataset['rain'][key]        
        data = CompressedRainData.load_from_raw(raw_data)           
        return self._ccrop.crop_domain(data)

    def _get_past_hourly_data(self, data_from_range_function, index):
        end_idx = self._input_end(index)
        output = []
        period = 6
        last_hour_data = None
        for _ in range(self._ilen):
            if end_idx <= 0:
                output.append(last_hour_data)
                continue

            start_idx = max(0, end_idx - period)
            data = data_from_range_function(range(start_idx, end_idx))
            last_hour_data = np.mean(data, axis=0, keepdims=True)
            output.append(last_hour_data)
            end_idx -= period

        return np.concatenate(output, axis=0)

    def _get_rain_data_from_range(self, index_range):
        rain = [self._ccrop(self._get_raw_rain_data(idx))[None, ...] for idx in index_range]
        rain = np.concatenate(rain, axis=0)
        rain[rain < 0] = 0
        rain = rain / RAIN_Q95
        return rain

    def _get_rain_data(self, index):
        # NOTE: average of last 5 frames is the rain data.
        return self._get_rain_data_from_range(self._input_range(index))
    
    ''' These should be refurbished later
    def _get_era5_data(self, index):
        index = [index + 6, index + 12, index + 18]
        keys = self._half_hour(index)
        era5 = [self._dataset['era5'][key][None,...] for key in keys]
        era5 = np.concatenate(era5, axis=0).astype(np.float32) # [3, 4, 20, 29, 23]
        return era5

    def _half_hour(self, idx):
        t = [self._time[i] for i in idx]
        counter = 0
        while counter < len(idx):
            if t[counter].minute >= 30:
                delta = 60 - t[counter].minute
                t[counter] = (lambda x: x + timedelta(minutes=delta))(t[counter])
            counter += 1
        return t
    '''

    def _get_most_recent_target(self, index, tavg_len=None):
        """
        Returns the averge rainfall which has happened in last self._tlen*10 minutes.
        """
        if tavg_len is None:
            tavg_len = self._tavg_len

        target_end_idx = self._input_end(index)
        target_start_idx = target_end_idx - tavg_len
        target_start_idx = max(0, target_start_idx)

        temp_data = [
            self._ccrop(self._get_raw_rain_data(idx))[None, ...] for idx in range(target_start_idx, target_end_idx)
        ]
        # print('Recent Target', list(range(target_start_idx, target_end_idx)))
        target = np.concatenate(temp_data, axis=0).astype(np.float32)
        target[target < 0] = 0
        assert target.shape[0] == tavg_len or target_start_idx == 0
        return target.mean(axis=0, keepdims=True)

    def _get_avg_target(self, index):
        target_start_idx = self._input_end(index) + self._toffset
        target_end_idx = target_start_idx + self._tlen * self._tavg_len
        temp_data = [
            self._ccrop(self._get_raw_rain_data(idx))[None, ...] for idx in range(target_start_idx, target_end_idx)
        ]
        # print('Target', list(range(target_start_idx, target_end_idx)))
        temp_data = np.concatenate(temp_data, axis=0)
        ''' no need average
        temp_data[temp_data < 0] = 0
        temp_data = temp_data / RAIN_Q95
        return temp_data # [18, 120, 120]
        '''
        ''' need to average'''
        target = []
        for i in range(self._tavg_len - 1, len(temp_data), self._tavg_len):
            target.append(temp_data[i - (self._tavg_len - 1):i + 1].mean(axis=0, keepdims=True))
        assert len(target) == self._tlen
        target = np.concatenate(target, axis=0)
        target[target < 0] = 0
        return target
        # return target/RAIN_Q95
        

    def __len__(self):
        return len(self._index_map) // self._sampling_rate

    def _get_internal_index(self, input_index):
        # If total we have 500 entries. Then with _ilen being 5, input index will vary in [0,100]
        input_index = input_index * self._sampling_rate
        index = self._index_map[input_index]
        return index

    def _get_past_hourly_rain_data(self, index):
        return self._get_past_hourly_data(self._get_rain_data_from_range, index)

    def _get_past_hourly_radar_data(self, index):
        return self._get_past_hourly_data(self._get_radar_data_from_range, index)

    def _random_perturbation(self, target):
        assert self._train is True

        def _rhs_idx(eps, N):
            return (eps, N) if eps > 0 else (0, N + eps)

        def _lhs_idx(eps, N):
            lidx = abs(eps) // 2
            ridx = N - (abs(eps) - lidx)
            return (lidx, ridx)

        Nx, Ny = target.shape[-2:]
        eps_x = int(np.random.normal(scale=self._random_std))
        eps_y = int(np.random.normal(scale=self._random_std))
        d_lx, d_rx = _rhs_idx(eps_x, Nx)
        d_ly, d_ry = _rhs_idx(eps_y, Ny)

        lx, rx = _lhs_idx(eps_x, Nx)
        ly, ry = _lhs_idx(eps_y, Ny)

        target[:, lx:rx, ly:ry] = target[:, d_lx:d_rx, d_ly:d_ry]
        target[:, :lx] = 0
        target[:, rx:] = 0
        target[:, :, :ly] = 0
        target[:, :, ry:] = 0
        return target

    def get_target_dt_and_season(self, index):
        index = range(index, index+6)
        target_dt = [self._time[i] for i in index] # 6 datetime
        dt_matrix = map(lambda x: self.periodization(x), target_dt)
        return np.array(list(dt_matrix), dtype = np.float32) # [6, 2]

    def periodization(self, inp_t: datetime):
        year = inp_t.year
        month = inp_t.month
        day = inp_t.day
        hour = inp_t.hour
        minute = inp_t.minute
        # arc_time = np.round(2 * np.pi * (hour + minute / 60) / 24, 10)
        arc_seas = np.round(2 * np.pi * 
        int(datetime(year, month, day).strftime("%j")) / int(datetime(year, 12, 31).strftime("%j")),
        10)
        # time_matrix = np.array([(np.sin(arc_seas), np.cos(arc_seas)),
        #                         (np.sin(arc_time), np.cos(arc_time))],)
        time_matrix = np.array([np.sin(arc_seas), np.cos(arc_seas)])
        return time_matrix
    
    def resize_as_target_input(self, dt_matrix: np.array, target:tuple) -> (np.array):
        '''
        transform into a uniform matrix
        dt_matrix shape = [N, season(2), time(2)]
        target = (H, W)
        '''
        new_matrix = np.ones([dt_matrix.shape[0], np.size(dt_matrix[0]), target[0], target[1]], dtype = np.float32)
        for t in range(dt_matrix.shape[0]):
                tmp = dt_matrix[t].reshape(-1)
                for j in range(len(tmp)):
                    new_matrix[t, j] = np.ones((target[0], target[1]), dtype=np.float32) * tmp[j]
        return new_matrix

    def initial_time(self, index):
        index = self._get_internal_index(index)
        index = index + 5 + self._toffset
        return self._time[index]

    def get_index_from_target_ts(self, ts):
        if ts in self._time:
            internal_index = self._time.index(ts)
            internal_index -= (self._toffset + self._ilen)
            index = self._index_map.index(internal_index)
            assert index % self._sampling_rate == 0
            return index // self._sampling_rate

        return None
    
    def dotProduct(self, ec_dir, slope_x, slope_y, index, keys:list = ['u', 'v']):
        # load ERA5
        dt = self._time[index + 5] # initial datetime
        filepath = os.path.join(ec_dir, str(dt.year), str(dt.year)+'{:02d}'.format(dt.month), 
                                f'era5_{dt.year}{dt.month:02}{dt.day:02}.nc'
                                )
        data = Dataset(filepath, 'r')
        avg_wind = []
        for key in keys:
            wind = data.variables[key][:] # masked array [24, 20, lat, lon]
            wind[np.where(wind.mask!=0)] = np.nan
            wind = np.nanmean(wind[:, -7], axis=(-1, -2)) # -7 is 850 hpa
            avg_wind.append(wind[dt.hour])       
        # dot
        result = slope_x * avg_wind[0] + slope_y * avg_wind[1]
        result = result.astype(np.float32)        
        return self._ccrop.crop_domain(result) # [..., 540, 420]

    def get_info_for_model(self):
        return {'input_shape': self[0][0].shape[2:]}

    def __getitem__(self, input_index):
        index = self._get_internal_index(input_index)
        input_data = []

        if self._dtype & DataType.Radar:
            #input_data.append(self._get_radar_data(index)) # numpy array [6, 21, 120, 120]
            input_data.append(self._get_radar_data(index)[:, None, ...]) # numpy array [6, 1, 540, 420]
            print('Radar', input_data[0].shape)

        if self._dtype & DataType.Elevation:
            input_data.append(self.six_multiplication(self._raw_altitude[0])[:, None]) # [6, 1, 540, 420]
            print('Elevation', input_data[-1].shape)

        if self._dtype & DataType.wtDot:
            result = self.dotProduct(ERA_DIR, self._slope_x, self._slope_y, index) # [120, 120]
            input_data.append(self.six_multiplication(result, factor=2000)[:, None]) # [6, 1, 540, 420]
            print('wtDot', input_data[-1].shape)

        if self._hourly_data:
            input_data.append(self._get_past_hourly_rain_data(index)[:, None, ...])
            input_data.append(self._get_past_hourly_radar_data(index)[:, None, ...])
            print('hourly', input_data[-1].shape)
            
        if self._dtype & DataType.Month:
            _time_data = self.get_target_dt_and_season(index) # [6, 2]
            input_data.append(self.resize_as_target_input(_time_data, self._img_size))# [6, 2, 120, 120]
            print('Month', input_data[-1].shape)

        # rain data must always be the last one
        if self._dtype & DataType.Rain:
            input_data.append(self._get_rain_data(index)[:, None, ...])
            print('Rain', input_data[-1].shape)          

        if len(input_data) > 1:            
            inp = np.concatenate(input_data, axis=1)
        else:
            inp = input_data[0]

        target = self._get_avg_target(index) 
        if self._train and self._random_std > 0:
            target = self._random_perturbation(target)

        # NOTE: mask needs to be created before tackling the residual option. We wouldn't know which entries are relevant
        # in the residual space.
        mask = np.zeros_like(target)
        mask[target > self._threshold] = 1

        #assert target.max() < 500
        # NOTE: There can be a situation where previous data is absent when self._tlen + self._tavg_len -1 > self._ilen
        if self._residual:
            assert self._random_std == 0
            recent_target = self._get_most_recent_target(index)
            target -= recent_target
            return inp, target, mask, recent_target
        
        return inp, target, mask

"""

#crop radar from yilin
if __name__ == '__main__':
    
    from tqdm import tqdm
    import numpy.ma as ma
    shape  = (NX, NY)
    croper = CropCenterNumpy(shape)
    vname  = "cv"

    diryilin = "/bk2/vancechen/CV/yilin"
    dircrop  = "/bk2/vancechen/CV/crop"

    for year in os.listdir(diryilin):
        year_dir = os.path.join(diryilin, year)
        for yearmonth in os.listdir(year_dir):
            ym_dir = os.path.join(year_dir, yearmonth)         
            for fname in tqdm(os.listdir(ym_dir)):
                data = Dataset(os.path.join(ym_dir,fname), 'r')
                radar = data.variables['cv'][:]
                crop_CV = croper.crop_domain(radar)
                out_name = os.path.join(dircrop, year, yearmonth, fname)
                save_nc(out_name, crop_CV, NX, NY, vname)

"""


