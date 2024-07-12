import os
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset

def load_nc(file_path:str) -> np.array:
    assert os.path.exists(file_path), f'{file_path} does not exist!'
    data = Dataset(file_path, 'r')
    Rainrate = data.variables['qperr'][:]
    Rainrate[np.where(Rainrate.mask!=0)] = -99
    Rainrate[Rainrate < 0] = 0
    lon = data.variables['lon'][:]
    lat = data.variables['lat'][:]
    assert np.shape(Rainrate) == (561, 441), f'{file_path} has a wrong shape'
    return np.array(Rainrate)

def save_nc(fname, data, vname='qperr'):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

    latStart = 20; latEnd = 27;
    lonStart = 118; lonEnd = 123.5;
    lat = np.linspace(latStart,latEnd,561)
    lon = np.linspace(lonStart,lonEnd,441)

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