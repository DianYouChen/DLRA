import os, pickle
import numpy as np
from datetime import datetime
from tqdm import tqdm

def load_from_raw(raw_data):
    data = -99 * np.ones((561, 441), dtype=np.float32)
    if len(raw_data) == 0:
        return data
    elif len(raw_data) == 3 and len(raw_data.shape) == 1:
        data[raw_data[0].astype(np.int32), raw_data[1].astype(np.int32)] = raw_data[2]
    else:
        data[raw_data[0, :].astype(np.int32), raw_data[1, :].astype(np.int32)] = raw_data[2, :]
    return data

def preprocess_data(data):
    d0, d1 = np.where(data != -99)
    sane_values = data[d0, d1]
    compressed_data = np.vstack([d0, d1, sane_values])
    return compressed_data

def load_from_raw_rain(raw_data):
    data = np.zeros((561, 441), dtype=np.float32)
    if len(raw_data) == 0:
        return data
    elif len(raw_data) == 3 and len(raw_data.shape) == 1:
        data[raw_data[0].astype(np.int32), raw_data[1].astype(np.int32)] = raw_data[2]
    else:
        data[raw_data[0, :].astype(np.int32), raw_data[1, :].astype(np.int32)] = raw_data[2, :]
    return data

def preprocess_data_rain(data):
    d0, d1 = np.where(data > 0)
    sane_values = data[d0, d1]
    compressed_data = np.vstack([d0, d1, sane_values])
    return compressed_data

def main():
    for pkl_name in tqdm(pkl_orig):
        new_rain = {}
        new_radar = {}
        with open(os.path.join(fpath, pkl_name), 'rb') as f:
            output = pickle.load(f) 
        keys = list(output.keys()) # three keys: radar, rain, era5
        for key in keys:
            if key == 'radar':
                subdata = output[key]
                for dt, value in subdata.items():
                    full_data = load_from_raw(value) # [561,441]
                    crop_data = full_data[325:445, 215:335] # [120,120]
                    comp_data = preprocess_data(crop_data)

                    new_radar[dt]=comp_data
                    del full_data, crop_data, comp_data
            if key == 'rain':
                subdata = output[key]
                for dt, value in subdata.items():
                    full_data = load_from_raw_rain(value) # [561,441]
                    crop_data = full_data[325:445, 215:335] # [120,120]
                    comp_data = preprocess_data_rain(crop_data)

                    new_rain[dt]=comp_data
                    del full_data, crop_data, comp_data
            if key == 'era5':
                subdata = output[key] # [dts][4, 20, 29, 23]
                new_era = subdata

        #output2 = {'radar': new_radar, 'rain': new_rain, 'era5': new_era}
        output2 = {'radar': new_radar, 'rain': new_rain}
        with open(os.path.join(output_dir, pkl_name), 'wb') as f:
            pickle.dump(output2, f)
        del output, output2
    
if __name__ == '__main__':
    fpath = './PKL_2Drd_rain'
    pkl_orig = os.listdir(fpath)
    pkl_orig.sort()
    output_dir = './PKL_2Dcrop_rain/'
    
    main()
