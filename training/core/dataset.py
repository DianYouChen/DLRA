import calendar
import os,sys
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool
from tqdm import tqdm
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")

os.environ['ROOT_DATA_DIR']='/bk2/vancechen/DLRA_database'
sys.path.extend(['/wk171/vancechen/training'])
from core.compressed_radar_data import CompressedRadarData
from core.compressed_rain_data import CompressedRainData
from core.constants import (DATA_PKL_DIR, RADAR_DIR, RAINFALL_DIR, CASES_WE_WANT,
                            SKIP_TIME_LIST, ERA_DIR)
from core.file_utils import RadarFileManager, RainFileManager
from core.time_utils import TimeSteps
# from core.specific_humidity import lookForNearestTime

from core.findSkipTime import findSkipTime_all, tooLargeRadar


def create_dataset(
        start_dt,
        end_dt,
        input_len,
        target_len,
        radar_dir=RADAR_DIR,
        rain_dir=RAINFALL_DIR,
        disjoint_entries=False,
):
    radar_fm = RadarFileManager(radar_dir)
    rain_fm = RainFileManager(rain_dir)
    cur_dt = start_dt
    dt_list = [cur_dt]
    while end_dt > cur_dt:
        cur_dt = TimeSteps.next(cur_dt)
        if cur_dt in SKIP_TIME_LIST:
            continue
        dt_list.append(cur_dt)

    dataset = []
    N = len(dt_list) - (input_len - 1) - target_len
    stepsize = 1
    if disjoint_entries:
        stepsize = input_len

    for i in range(0, N, stepsize):
        inp_radar_fpaths = [radar_fm.fpath_from_dt(dt) for dt in dt_list[i:i + input_len]]
        inp_rain_fpaths = [rain_fm.fpath_from_dt(dt) for dt in dt_list[i:i + input_len]]
        inp = list(zip(inp_radar_fpaths, inp_rain_fpaths))

        target = [rain_fm.fpath_from_dt(dt) for dt in dt_list[i + input_len:i + input_len + target_len]]
        dataset.append((inp, target))

    print(f"Created Dataset:{start_dt.strftime('%Y%m%d_%H%M')}-{end_dt.strftime('%Y%m%d_%H%M')}, "
          f" Disjoint:{int(disjoint_entries)} InpLen:{input_len} TarLen:{target_len} {len(dataset)}K points")

    return dataset


def keep_first_half(dic):
    """
    Keep data only for first 15 days of each month.
    """
    output = {k: v for k, v in dic.items() if k.day <= 15}
    return output


def keep_later_half(dic):
    """
    Keep only that data which is not being kept in keep_first_half()
    """
    fh_dict = keep_first_half(dic)
    return {k: v for k, v in dic.items() if k not in fh_dict}

def last_day_of_month(any_day):
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, 
    # or programattically said, the previous day of the first of next month
    return next_month - timedelta(days=next_month.day) + timedelta(hours=23, minutes=50)

def load_from_list(dradar_test, drain_test, dt_pkl:list, resume_list:list) -> (dict):
    year = 0; month = 0
    for dt in resume_list:
        if dt not in dt_pkl:
            if (year != dt.year) or (month != dt.month):
                year = dt.year
                month = dt.month
                start_dt = datetime(year, month, 1)
                end_dt = last_day_of_month(start_dt)
                fname = os.path.join(DATA_PKL_DIR, 'AllDataDict_{start}_{end}.pkl')
                fname = fname.format(start=start_dt.strftime('%Y%m%d-%H%M'),
                                     end=end_dt.strftime('%Y%m%d-%H%M'),)
                with open(fname, 'rb') as f:
                    output = pickle.load(f)
            try:
                dradar_test[dt] = output["radar"][dt]
                drain_test[dt] = output["rain"][dt]
            except:
                # if dt isn't in output["radar"]
                continue
        
def load_data(start_dt, end_dt, is_validation=False, is_test=False, is_train=False, workers=0, missing_dt=[]):
    assert int(is_validation) + int(is_test) + int(is_train) == 1, 'Data must be either train, test or validation'
    dtype_str = ['Train'] * int(is_train) + ['Validation'] * int(is_validation) + ['Test'] * int(is_test)
    print(f'[Loading {dtype_str[0]} Data] {start_dt} {end_dt}')

    arguements = []
    assert start_dt < end_dt
    cur_dt = start_dt
    while cur_dt < end_dt:
        # Ashesh想要一個月一個月的存資料
        last_day_month = calendar.monthrange(cur_dt.year, cur_dt.month)[1]
        # NOTE: 23:50 is the last event. this may change if we change the granularity
        offset_min = 23 * 60 + 50 - (cur_dt.hour * 60 + cur_dt.minute)
        cur_end_dt = min(end_dt, cur_dt + timedelta(days=last_day_month - cur_dt.day, seconds=60 * offset_min))
        arguements.append((cur_dt, cur_end_dt))
        cur_dt = TimeSteps.next(cur_end_dt)
    #print('Months I want are:', arguements)

    
    data_dicts = []
    if workers > 0:
        with Pool(processes=workers) as pool:
            with tqdm(total=len(arguements)) as pbar:
                for i, data_dict in enumerate(pool.imap_unordered(_load_data, arguements)):
                    if data_dict.get('fpath'):
                        pbar.set_description(f"Loaded from {data_dict['fpath']}")
                    pbar.update()
                    data_dicts.append(data_dict)
    else:
        for args in tqdm(arguements):
            data_dicts.append(_load_data(args, missing_dt))
    
    radar_dict = {}
    rain_dict = {}
    for d in data_dicts:
        d = comb_n_kick(d, SKIP_TIME_LIST, CASES_WE_WANT, test=is_test, dev=is_validation, train=is_train)
        radar_dict = {**radar_dict, **d['radar']}
        rain_dict = {**rain_dict, **d['rain']}
    if is_test:
        # load the missing cases we want
        dt_pkl = list(radar_dict.keys())
        load_from_list(radar_dict, rain_dict, dt_pkl, CASES_WE_WANT)
    assert len(radar_dict.keys()) == len(rain_dict.keys())
    return {'rain': rain_dict, 'radar': radar_dict}

def _load_data(args, Missing=[]):
    start_dt, end_dt = args
    fname = os.path.join(DATA_PKL_DIR, 'AllDataDict_{start}_{end}.pkl')
    fname = fname.format(
        start=start_dt.strftime('%Y%m%d-%H%M'),
        end=end_dt.strftime('%Y%m%d-%H%M'),
    )
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            output = pickle.load(f)
        # output['fpath'] = fname
        return output

    radar_fm = RadarFileManager(RADAR_DIR, compressed=True)
    rain_fm = RainFileManager(RAINFALL_DIR, compressed=True)
    cur_dt = start_dt
    dt_list = []

    while end_dt >= cur_dt:
        if cur_dt not in Missing:
            dt_list.append(cur_dt)

        cur_dt = TimeSteps.next(cur_dt)

    radar_data = {}
    rain_data = {}
    for dt in dt_list:
        radar_data[dt] = CompressedRadarData(radar_fm.fpath_from_dt(dt)).load_raw() # [x, y, value][3, n]
        rain_data[dt] = CompressedRainData(rain_fm.fpath_from_dt(dt)).load_raw(can_raise_error=True)

    output = {'radar': radar_data, 'rain': rain_data}
    with open(fname, 'wb') as f:
        pickle.dump(output, f)

    return output

def comb_n_kick(data, remove_all:list, leave_alone:list, test=False, dev=False, train=False):
    if train:
        dt_pkl = list(data['radar'].keys())
        for elimi in remove_all:
            if elimi in dt_pkl:
                del data['radar'][elimi]
                del data['rain'][elimi]
    elif dev: # first-half month for valid
        data = {'radar': keep_first_half(data['radar']), 'rain': keep_first_half(data['rain'])}
        dt_pkl = list(data['radar'].keys())
        for elimi in remove_all:
            if elimi in dt_pkl:
                del data['radar'][elimi]
                del data['rain'][elimi]
    elif test: # last-half month for valid
        data = {'radar': keep_later_half(data['radar']), 'rain': keep_later_half(data['rain'])}
        dt_pkl = list(data['radar'].keys())
        for elimi in remove_all:
            if (elimi in dt_pkl) & (elimi not in leave_alone):
                del data['radar'][elimi]
                del data['rain'][elimi]
    return data

if __name__ == '__main__':

    from netCDF4 import Dataset

    """
    #%% skip time for 2019.01-2020.10
    # missing data
    # must ends with year
    filepath_radar = '/bk2/vancechen/CV/raw/radar_compressed/2022'
    filepath_rain = '/bk2/vancechen/QPESUMS/raw/rain_compressed/2022'

    missing = findSkipTime_all(filepath_radar, datetime(2022,8,1), datetime(2022,8,31,23,50))
    missing += findSkipTime_all(filepath_rain, datetime(2022,8,1), datetime(2022,8,31,23,50))

    # too large number @ radar
    missing += tooLargeRadar(filepath_radar, workers=8)

    missing = sorted(list(set(missing)))
    ###
    start_dt = datetime(2022, 8, 1)
    end_dt = datetime(2022, 8, 31, 23, 50)
    load_data(start_dt, end_dt, is_test=True, workers=0, missing_dt=missing)

    """

    error_keys = []

    for mm in range(7,8):

        start_dt = datetime(2022, mm+1, 1)
        #end_dt = datetime(2022, mm+1, 31, 23, 50)
        end_dt = start_dt + relativedelta(months=1) - timedelta(minutes=10)

        fname = os.path.join('/bk2/vancechen/DLRA_database/PKL_2Drd_rain10m', 'AllDataDict_{start}_{end}.pkl')
        fname = fname.format(
                start=start_dt.strftime('%Y%m%d-%H%M'),
                end=end_dt.strftime('%Y%m%d-%H%M'),
                )
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                output = pickle.load(f)
                time = list(output['rain'].keys())
                for k in range(len(time)):
                    xxx = output['radar'][time[k]]
                    if xxx.max() > 1e+9:
                        error_keys.append(time[k])
    print(error_keys)

    """
    fpath = os.path.join("/bk2/vancechen/DLRA_database/radar_error","nc_files")
    fname = os.path.join(fpath, "date_errors"+".pkl")

    with open(fname, 'wb') as f:
        pickle.dump(error_keys, f)
    """

    
    """
    with open('/bk2/vancechen/DLRA_database/PKL_2Drd_rain10m/AllDataDict_20220701-0000_20220731-2350.pkl', 'rb') as f:
        output = pickle.load(f)

    my_key = list(output['rain'].keys())
    xxx = output['radar'][my_key[400]]
    print((my_key)[400])
    print(xxx)
    
    """

    
    
