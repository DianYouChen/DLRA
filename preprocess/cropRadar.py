import os, argparse
import netCDF4 as nc
import numpy as np
from tqdm import tqdm

def cutRadar(mother_dir):
    ym_paths = os.listdir(mother_dir)
    ym_paths.sort()
    for ym_path in tqdm(ym_paths):
        files = os.listdir(os.path.join(mother_dir, ym_path))
        files.sort()
        for file in files:
            # file name must be like 20210601_0020.nc
            # narrow the region
            ds = nc.Dataset(os.path.join(mother_dir, ym_path, file))
            cv = ds['cv'][:] #881*921
            lat = ds['lat'][:] #881
            lon = ds['lon'][:] #921
            if np.shape(cv) == (561, 441):
                print(f'{file} is already cropped.')
                continue
            else:
                assert np.shape(cv) == (881, 921), f'{file} has a wrong shape.'
                os.remove(os.path.join(mother_dir, ym_path, file))

                f = nc.Dataset(os.path.join(mother_dir, ym_path, file),'w',format = 'NETCDF4')
                f.createDimension('lat',561)   
                f.createDimension('lon',441)
                f.createVariable('cv',np.float32,('lat', 'lon')) 
                f.createVariable('lat',np.float32,('lat'))  
                f.createVariable('lon',np.float32,('lon'))
                f.variables['lat'][:] = lat[160:721] # lat: 720-160+1=561
                f.variables['lon'][:] = lon[240:681] # lon: 680-240+1=441
                f.variables['cv'][:] = cv[160:721, 240:681]
                f.close()
                print(f'{file} is done.')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('radar_input_path', type=str, help='radar source directory.')
    args = parser.parse_args()
    
    cutRadar(args.radar_input_path)