import os
from datetime import datetime, timedelta
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def findSkipTime_all(filepath, s:datetime, e:datetime):
    # ideal day hour minutes
    ideal_t = []
    cur_dt = s
    while cur_dt <= e: 
        ideal_t.append(cur_dt)
        cur_dt += timedelta(minutes=10)
    
    # reality
    restore_time = []
    year_months = os.listdir(filepath)
    year_months.sort()
    for y_m in year_months:
        y_m_d = os.listdir(os.path.join(filepath, y_m))
        y_m_d.sort()
        for day in y_m_d:
            d_h_m_file = os.listdir(os.path.join(filepath, y_m, day))
            d_h_m_file.sort()
            for file1 in d_h_m_file:
                if file1.endswith('.nc.gz'):
                    # 20210102_2330.nc.gz
                    file = file1.split('.')[0]
                    year = int(file[:4])
                    month = int(file[4:6])
                    day = int(file[6:8])
                    hour = int(file[9:11])
                    minute = int(file[11:13])
                if file1.startswith('MREF'):
                    # MREF3D21L.20150101.2350.gz
                    file = file1.split('.')[1]+'_'+file1.split('.')[2]
                    year = int(file[:4])
                    month = int(file[4:6])
                    day = int(file[6:8])
                    hour = int(file[9:11])
                    minute = int(file[11:13])
                d_h_m = datetime(year, month, day, hour, minute)
                restore_time.append(d_h_m)
    
    # double check
    loss_dt = []
    for i in ideal_t:
        if i not in restore_time:
            loss_dt.append(i)
    print('check miss data from:', filepath)
    print('lost numbers:', len(loss_dt))
    return loss_dt

def noname(file, filepath, threshold=10000):
    filename = os.path.join(filepath, file)
    # 20210102_2330.nc.gz
    data = np.loadtxt(filename)
    if len(data) == 0:
        return 
    if np.max(data) > threshold:
        year = int(file[:4])
        month = int(file[4:6])
        dayy = int(file[6:8])
        hour = int(file[9:11])
        minute = int(file[11:13])
        d_h_m = datetime(year, month, dayy, hour, minute)
        return d_h_m
    
def tooLargeRadar(filepath, workers=8):
    restore_time = []
    
    year_months = os.listdir(filepath)
    year_months.sort()
    pbar = tqdm(year_months)
    for y_m in pbar:
        pbar.set_description(f'check {y_m}')
        y_m_d = os.listdir(os.path.join(filepath, y_m))
        y_m_d.sort()
        for day in y_m_d:
            args=[]
            for d_h_m_file in os.listdir(os.path.join(filepath, y_m, day)):
                y_dir = os.path.join(filepath, y_m, day)
                args.append((d_h_m_file, y_dir))
            with Pool(processes=workers) as pool:
                restore_time.extend(pool.starmap(noname, args))
    print('too large radar numbers:', len(list(filter(None, restore_time))))          
    return list(filter(None, restore_time))

if __name__ == '__main__':
    filepath_radar = '/wk171/handsomedong/docker_eval/input_d/radar2D_compressed/2021'
    filepath_rain = '/wk171/handsomedong/docker_eval/input_d/rain_compressed/2021'
    
    start = datetime(2021,1,1)
    end = datetime(2021,10,31,23,50)