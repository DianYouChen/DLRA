import argparse
import os, sys
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from raw_data import RawRainData, RawRadarData

class DataCompressor:
    def __init__(self, source_dir:str, destination_dir:str, overwrite:bool=False, year_id:int=1):
        # overwirte: If True, replace the older files.
        # sdir: /.../yyyy/yyyymm/yyyymmdd_hhmm.nc
        self.sdir = source_dir
        self.ddir = destination_dir
        self.overwrite = overwrite
        self.id = year_id
        if not os.path.exists(self.ddir):
            os.mkdir(self.ddir)
        if year_id == 0:
            self.compress_all = self.compress_all_0
        elif year_id == 1:
            self.compress_all = self.compress_all_1
        print(f'[{self.__class__.__name__}] SRC:{self.sdir} DST:{self.ddir} Overwrite:{self.overwrite}')
        
    def compress_all_0(self, workers:int=1):
        tmp = os.listdir(self.sdir)
        tmp.sort()
        tqdm_inst = tqdm(tmp)
        for year_month in tqdm_inst:
            tqdm_inst.set_description(f'Processing {year_month}')
            ym_path = os.path.join(self.sdir, year_month)
            ### only rain from 2014-2018 needed
            ymd_path = [os.path.join(ym_path, x) for x in os.listdir(ym_path)]
            ymd_path.sort()
            for ymd in ymd_path:
                files = os.listdir(ymd)
                files.sort()
                arguments = []
                for file in files:
                    arguments.append((file, ymd))
                with Pool(processes=workers) as pool:
                    pool.starmap(self.compress_one, arguments)
                    
    def compress_all_1(self, workers:int=1):
        tmp = os.listdir(self.sdir)
        tmp.sort()
        tqdm_inst = tqdm(tmp)
        for year_month in tqdm_inst:
            tqdm_inst.set_description(f'Processing {year_month}')
            ym_path = os.path.join(self.sdir, year_month)
            files = os.listdir(ym_path)
            files.sort()
            arguments = []
            for file in files:
                arguments.append((file, ym_path))
            with Pool(processes=workers) as pool:
                pool.starmap(self.compress_one, arguments)
            

    def compress_one(self, fname: str, ym_path: str):
        fpath = os.path.join(ym_path, fname)
        
        if 'radar' in fpath:
            loader = RawRadarData(fpath)
            data = loader.load()['radar']
        elif 'rain' in fpath:
            loader = RawRainData(fpath)
            data = loader.load()['rain']
        dt = loader.datetime()
        if self.overwrite is False and os.path.exists(self.target_path(dt, fname)):
            return
        self.create_dir(dt)
        self.save(data, dt, fname)
#         print(f'{fname} is done!')
    
    def target_path(self, dt: datetime, fname: str):
        return os.path.join(self.ddir,
                            str(dt.year),
                            f'{dt.year}{dt.month:02}',
                            f'{dt.year}{dt.month:02}{dt.day:02}',
                            fname + '.gz'
                           )  
    
    def create_dir(self, dt: datetime):
        day_dir = os.path.join(self.ddir,
                               str(dt.year),
                               f'{dt.year}{dt.month:02}',
                               f'{dt.year}{dt.month:02}{dt.day:02}',
                              )
        os.makedirs(day_dir, exist_ok=True)

    def save(self, data: np.array, dt: datetime, fname: str):
        fpath = self.target_path(dt, fname)
        np.savetxt(fpath, self.preprocess_data(data), fmt='%1.3f')

    def preprocess_data(self, data):
        d0, d1 = np.where(data > 0)
        sane_values = data[d0, d1]
        compressed_data = np.vstack([d0, d1, sane_values])
        return compressed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='source directory. It should contain year_month as subdirectories')
    parser.add_argument('dest', type=str, help='destination directory')
    parser.add_argument('--year_id', type=int, default=1, help='2014-2018: 0; 2019-2021: 1')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    comp = DataCompressor(args.src, args.dest, overwrite=args.overwrite, year_id=args.year_id)
    comp.compress_all(workers=args.workers)