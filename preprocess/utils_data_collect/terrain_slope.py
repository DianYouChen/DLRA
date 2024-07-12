import os
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from netCDF4 import Dataset

class load_shp:
    def __init__(self, filename):
        self._fname = filename
        self.load()

    def load(self):
        self._df = gpd.read_file(self._fname)
        # print(self._df.head(3))

    def getMap(self, target:str="坡度"):
        # lat&lon from shapefile
        # total len = 45264 = 276x164
        lat_shp = self.getAttr("N_1") 
        lon_shp = self.getAttr("E_1")
        ele_shp = self.getAttr(target)
        total_len = len(ele_shp)

        latList = sorted(list(set(lat_shp))) # 276
        lonList = sorted(list(set(lon_shp))) # 164
        print(f'Latitude of Map: {latList[0]}~{latList[-1]},'\
              f'Longitude of Map: {lonList[0]}~{lonList[-1]}')
        ele_map = np.zeros([len(latList), len(lonList)], dtype=np.float32)

        for i in tqdm(range(total_len)):
            ele_map[np.where(latList == lat_shp[i]), np.where(lonList == lon_shp[i])] = ele_shp[i]
        return ele_map, latList, lonList

    def getAttr(self, attName:str):
        data_array = self._df[attName].to_numpy()
        return list(map(lambda x: np.around(x, 4), data_array))

def crop_lat_lon(initial_lat:np.array, initial_lon:np.array, nlat, nlon):
    start_lat = initial_lat.shape[0] // 2 - nlat // 2
    start_lon = initial_lon.shape[0] // 2 - nlon // 2
    target_lat  = initial_lat[start_lat:start_lat+nlat]
    target_lon  = initial_lon[start_lon:start_lon+nlon]
    return target_lat, target_lon

def mapping(lat_list:list, lon_list:list, input_map:np.array, target_shape):
    x = len(lon_list); y = len(lat_list)
    assert np.shape(input_map) == (y, x), "The shape of map is not consistent with lat/lon list."
    x_target = target_shape[1]; y_target = target_shape[0]

    # lat&lon of cropped TW [540x420]
    latStart = 20; latEnd = 27
    lonStart = 118; lonEnd = 123.5
    qpesums_lat = np.linspace(latStart,latEnd,561)
    qpesums_lon = np.linspace(lonStart,lonEnd,441)
    lat_crop, lon_crop = crop_lat_lon(qpesums_lat, qpesums_lon, y_target, x_target)

    print(f'Latitude Range: {lat_crop.min()}~{lat_crop.max()}, '\
          f'Longitude Range: {lon_crop.min()}~{lon_crop.max()} '\
          f'for cropped TW [{y_target}x{x_target}]')

    # Notice that the cropped lat/lon is gonna exceed the range of input lat/lon list.
    loc_lat = np.argmin(np.abs(np.array(lat_list) - lat_crop[0]))
    loc_lon = np.argmin(np.abs(np.array(lon_list) - lon_crop[0]))

    # the entire cropped Taiwan is in input_map
    if (loc_lat+y_target-1 < y) & (loc_lon+x_target-1 < x):
        target_range = input_map[loc_lat:loc_lat+y_target, loc_lon:loc_lon+x_target]
    # the x direction of cropped Taiwan exceeds the input_map
    elif (loc_lat+y_target-1 < y) & (loc_lon+x_target-1 >= x):
        target_range = np.zeros([y_target, x_target], dtype=np.float32)
        target_range[:, :x-loc_lon] = input_map[loc_lat:loc_lat+y_target, loc_lon:]
    # the y direction of cropped Taiwan exceeds the input_map
    elif (loc_lat+y_target-1 >= y) & (loc_lon+x_target-1 < x):
        target_range = np.zeros([y_target, x_target], dtype=np.float32)
        target_range[:y-loc_lat, :] = input_map[loc_lat:, loc_lon:loc_lon+x_target]
    # both x and y direction are out of input_map
    else:
        target_range = np.zeros([y_target, x_target], dtype=np.float32)
        start_lat = y_target//2-y//2+9; start_lon = x_target//2-x//2+19
        target_range[start_lat:start_lat+y, start_lon:start_lon+x] = input_map[loc_lat:, loc_lon:]

    return target_range, lat_crop, lon_crop

class dotProduct:
    '''
    This function aims to do 2 things.
    1. convert the vector direction from -Gradient(point to small) to Gradient(point to large)
    2. convert the meteorological coor. into mathematical coor.
    3. use two vectors(wind & slope) within mathematical coor. to calculate dot product

    In Chinese, 從風花圖的向量(270度是西風)先轉成u/v風場(90度才是西風),再從u/v風場轉換成數學角(0度才是西風).
    同理用在地形, 坡向是下坡方向(風花圖),先轉成上坡方向(gradient, u/v風場), 再轉成數學角.
    最後對兩者做內積.

    hodograph: is a 2D numpy array, 速端曲線
    '''
    def __init__(self, hodo_data:np.array, ec_dir):
        self._data = hodo_data
        self._ec_dir = ec_dir

    def hodo2uv(self, matrix):
        # +180 turn out to be u/v field; hodo Coor. to u/v Coor.
        matrix = matrix + 180
        matrix[matrix>=360] = matrix[matrix>=360] - 360
        return matrix

    def uv2math(self, matrix):
        # symmetrize against 45-225 deg line; u/v Coor. to math Coor.
        matrix = 90 - matrix + 360
        matrix[matrix>=360] = matrix[matrix>=360] - 360
        return matrix

    def forward(self, is_elevation=False):
        if is_elevation:
            # self-calculate x & y components from given elevation
            y_slope, x_slope = self.calSlope()
            return x_slope, y_slope
        else:
            # manufacturing x & y components from given aspect
            aspect_degree = self.uv2math(self.hodo2uv(self._data))
            aspect_radian = np.pi * (aspect_degree / 180)
            aspect_x, aspect_y = self.Trigonometry(aspect_radian)
            return aspect_x, aspect_y

    def Trigonometry(self, radian, norm=1):
        # assume slope == 1
        return (norm * np.cos(radian), norm * np.sin(radian))

    def calSlope(self):
        # For dim=1, idx 0 is south; idx -1 is north
        ns_shift = np.zeros([self._data.shape[0], self._data.shape[1]])
        ew_shift = np.zeros([self._data.shape[0], self._data.shape[1]])
        ns_shift[:-1] = self._data[1:]
        ew_shift[:, :-1] = self._data[:, 1:]
        ns_slope = -(self._data - ns_shift) # north - south
        ew_slope = -(self._data - ew_shift) # east - west
        return ns_slope, ew_slope
    
    def blurness(self, data, k_size=3):
        # Moving Average
        # data shape = [H, W]
        pd = k_size//2
        tmp = np.pad(data, ((pd,pd), (pd,pd)), 'constant')
        tmpp = np.copy(tmp)
        for i in range(pd, tmp.shape[0]-pd):
            for j in range(pd, tmp.shape[1]-pd):
                tmpp[i, j] = tmp[i-pd:i+pd+1, j-pd:j+pd+1].mean()
        return tmpp[pd:-pd, pd:-pd]

    def __call__(self):
        '''
        read from PKL, output PKL includes terrainXwind
        '''
        # get terrain slope
        x_slope, y_slope = self.forward(is_elevation=True)

        # read EC u/v wind
        file_list = self.list_ec()
        for file in tqdm(file_list):
            u_wind, v_wind = self.load_ec(file, ['u', 'v'])
            # dot = u * aspect_x + v * aspect_y -> [24, 120, 120]
            # write into nc
            # f = h5py.File(os.environ['ROOT_DATA_DIR']+'/output_data_v2.nc',mode='w')
            # dset1 = f.create_dataset(name = 'output',data = tmp_output,compression = 'gzip')
            # dset2 = f.create_dataset(name = 'target',data = tmp_target,compression = 'gzip')
            # f.close()

            # f = nc.Dataset(os.path.join(OUTPUT_DIR,f'{tsave[:6]}',f'DLRA_{tsave}.nc'),
            #                'w',
            #                format = 'NETCDF4')
            # f.createDimension('lat',561)   
            # f.createDimension('lon',441)
            # f.createDimension('fcst_hr',3)
            # f.createVariable('rr',np.float32,('fcst_hr','lat', 'lon'),compression='zlib') 
            # f.createVariable('rr_3',np.float32,('lat', 'lon'),compression='zlib')
            # f.createVariable('lat',np.float32,('lat'),compression='zlib')  
            # f.createVariable('lon',np.float32,('lon'),compression='zlib')
            # f.variables['lat'][:] = np.linspace(20,27,561)
            # f.variables['lon'][:] = np.linspace(118,123.5,441)
            # f.variables['rr'][:] = output_large
            # f.variables['rr_3'][:] = output_3
            # f.close()

    def load_ec(self, fileName, keys:list = ['u', 'v']):
        data = Dataset(fileName, 'r')
        output = []
        for key in keys:
            wind = data.variables[key][:] # masked array
            wind[np.where(wind.mask!=0)] = np.nan
            wind = np.nanmean(wind[:, -1], axis=(-1, -2))
            output.append(wind)
        return output

    def list_ec(self):
        file_repo = []
        years = self.hierarchical(self._ec_dir)
        for year in years:
            months = self.hierarchical(year)
            for month in months:
                day_files = self.hierarchical(month)
                file_repo.extend(day_files)
        return file_repo

    def hierarchical(self, parent_dir):
        children = sorted(os.listdir(parent_dir))
        children = [os.path.join(parent_dir, child) for child in children]
        return children