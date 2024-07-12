import os, json
from typing import Tuple
import numpy as np
from datetime import datetime, timedelta

from util.raw_data import load_nc, save_nc
from util.file_util import  get_latest
from util.array_data import FixedSizeArray

"""
Split 1-h accumulated rainfall (mm) into 10-m rain rate (mm/h)

"""

class Cleaver:
    def __init__(self, inp_dir:str, oup_dir:str, cwd_dir:str, 
                 mask_fname:str, fixed_array_fname:str):
        self.inp_dir = inp_dir
        self.oup_dir = oup_dir
        self.cwd_dir = cwd_dir
        self.mask_path = os.path.join(cwd_dir, mask_fname)
        self.fixed_size_array_path = os.path.join(cwd_dir, fixed_array_fname)

        # get current file with full path
        self.curr_fname = get_latest(inp_dir)
        self.curr_dt = self.get_time_from_path(self.curr_fname)
        self.curr_data = load_nc(self.curr_fname)

    def get_time_from_path(self, full_path:str) -> datetime:
        # full_path = ../data/output/qperr/2022/202209/20220918_0100.nc
        dt_str = os.path.basename(full_path)
        return datetime.strptime(dt_str.split('.')[0], "%Y%m%d_%H%M")

    @classmethod
    def get_path_from_time(cls, 
                           root_path:str, 
                           dt:datetime, 
                           file_format:str = "%Y%m%d_%H%M.nc"
                           ) -> str:
        year = dt.year
        month = dt.month
        return os.path.join(root_path, 
                            f"{year}", 
                            f"{year}{month:02d}", 
                            dt.strftime(file_format)
                            )

    def check(self):
        """
        check if the previous data exists 
        """
        prev_time = self.curr_dt - timedelta(minutes=10)
        output_dt = self.curr_dt - timedelta(minutes=50)
        output_fname = Cleaver.get_path_from_time(self.oup_dir, output_dt)
        if os.path.exists(output_fname):
            print('=== no file updates, skip temporal downscaling ===')
            return

        if (not os.path.exists(Cleaver.get_path_from_time(self.inp_dir, prev_time))) \
            or (not os.path.exists(self.mask_path)) \
            or (not os.path.exists(self.fixed_size_array_path)):
            # 1. build Mask
            if os.path.exists(self.mask_path):
                os.remove(self.mask_path)
            mask = (self.curr_data == 0) * 1
            self.write_mask(mask)
            
            # 2. build fix_sized_array
            if os.path.exists(self.fixed_size_array_path):
                os.remove(self.fixed_size_array_path)
            fix_sized_array = FixedSizeArray()
            fix_sized_array.fit_mask(mask)
            self.write_fixed_size_array(fix_sized_array.data)

            # 3. save t=-6 nc file
            last_data = fix_sized_array.data[0]
            save_nc(output_fname, last_data)
            print(f'10-min data: {os.path.basename(output_fname)} established!')

            # 4. leave
            print('No previous data for splitting, first init.')
            
        else:
            # 1. read Mask, read fix_sized_array(剔除idx=0)
            mask_orig = self.read_mask()
            fix_sized_array = FixedSizeArray(pre_data=self.fixed_size_array_path)
            
            # 2. (data - (fix_sized_array[5]) * mask
            data_increment = (self.curr_data - fix_sized_array.five_sum()) * mask_orig
            data_increment[data_increment < 0] = 0

            # 3. push #2 into fix_sized_array[6]
            fix_sized_array.append(data_increment)

            # 4. find data == 0
            mask_new = (self.curr_data == 0) * 1
            
            # 5. clean fix_sized_array[6] & build
            fix_sized_array.fit_mask(mask_new)
            self.write_fixed_size_array(fix_sized_array.data)

            # 6. update mask & build
            mask_new += mask_orig
            mask_new[mask_new > 1] = 1
            self.write_mask(mask_new)

            # 7. save t=-6 nc file
            last_data = fix_sized_array.data[0]
            save_nc(output_fname, last_data)
            print(f'10-min data: {os.path.basename(output_fname)} established!')

            # 8. leave
            print(f'{os.path.basename(self.curr_fname)} has been downscaled.')
    
    def read_mask(self) -> np.ndarray:
        with open(self.mask_path, "r") as f:
            info = json.load(f)
    
        if info['store'] == 'Ace':
            return np.ones((561, 441), dtype=np.int16)
        elif info['store'] == 'False':
            mask_orig = np.ones((561, 441), dtype=np.int16)
            for pair in info["metrix"]:
                mask_orig[pair[0], pair[1]] = 0
        elif info['store'] == 'True':
            mask_orig = np.zeros((561, 441), dtype=np.int16)
            if len(info["metrix"]) == 0:
                return mask_orig
            for pair in info["metrix"]:
                mask_orig[pair[0], pair[1]] = 1
        return mask_orig

    def write_mask(self, mask) -> None:
        location_list, hint = self.decide_forth_or_back(mask)
        information = {
            "store": hint,
            "metrix": location_list
        }
        # Writing to .json
        with open(self.mask_path, "w") as outfile:
            json.dump(information, outfile, indent=2)

    def decide_forth_or_back(self, mask) -> Tuple[list, str]:
        total_size = np.size(mask)
        activate_num = np.sum(mask == 1)
        output = []
        
        if activate_num == total_size:
            return output, 'Ace'

        if activate_num >= total_size/2:
            # 存未啟動的格點位置
            d0, d1 = np.where(mask == 0)
            for x, y in zip(d0, d1):
                output.append((int(x), int(y)))
            return output, 'False'
        else:
            # 存已啟動的格點位置
            d0, d1 = np.where(mask == 1)
            for x, y in zip(d0, d1):
                output.append((int(x), int(y)))
            return output, 'True'
    
    def write_fixed_size_array(self, data):
        # data = [6, 561, 441]
        information = {}
        for i in range(data.shape[0]):
            output = []
            d0, d1 = np.where(data[i] != 0)
            sane_values = data[i][d0, d1]
            for x, y, z in zip(d0, d1, sane_values):
                output.append((int(x), int(y), float(z)))
            information[i] = output
        # Writing to .json
        with open(self.fixed_size_array_path, "w") as outfile:
            json.dump(information, outfile, indent=2)

if __name__ == "__main__":
    inp_dir = "/bk2/vancechen/QPESUMS/yilin"
    oup_dir = "/bk2/vancechen/QPESUMS/raw"
    cwd_dir = "/wk171/vancechen/cleaver"
    mask_fname = 'mask.json'
    fixed_array_fname = 'fixedSizeArray.json'

    # instantiate Cleaver object, all of the executions will be done via this object
    rain_cleaver = Cleaver(inp_dir, oup_dir, cwd_dir, mask_fname, fixed_array_fname)
    rain_cleaver.check()