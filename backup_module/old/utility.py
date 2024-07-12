import numpy as np
import pickle
from tqdm import tqdm

class performance():
    def __init__(self, 
                 prediction, 
                 groundTruth, 
                 threshold=[1,3,5,10,15,20,30,40],
                ):
        self._pred = prediction
        self._gdth = groundTruth
        self._trhd = threshold
        self._CSI, self._HSS = self.calScore()
        #print(f'calScore finished.\ttotal data number: {self._pred.shape[0]}')
    
    def calScore(self):
        return self.calScore_deep(self._pred, self._gdth, self._trhd)
    
    def calScore_deep(self, pred, ground, thresholds):
        assert pred.shape == ground.shape
        thresholds = np.array(thresholds, dtype=np.int32)
        CSI = np.array([])
        HSS = np.array([])
        for threshold in thresholds:
            # numbers
            hits = np.sum(pred[ground>=threshold]>=threshold)
            misses = np.sum(pred[ground>=threshold]<threshold)
            false_alarms = np.sum(pred[ground<threshold]>=threshold)
            correct_negatives = np.sum(pred[ground<threshold]<threshold)
            # Since the nan ratio of the pred is quite small, replacing total grids with sum of all terms is acceptable.
            #assert hits+misses+false_alarms+correct_negatives == np.size(pred)
            sample_size = hits+misses+false_alarms+correct_negatives
            # fct scores
            CSI_tmp = hits/(hits+misses+false_alarms)
            expected_correct = ((hits + misses)*(hits + false_alarms)+
                                (correct_negatives + misses)*(correct_negatives + false_alarms))/sample_size
            HSS_tmp = (hits+correct_negatives-expected_correct)/(sample_size-expected_correct)

            CSI = np.append(CSI, CSI_tmp)
            HSS = np.append(HSS, HSS_tmp)
            del CSI_tmp, HSS_tmp
        return CSI, HSS
    
    def showCalResult(self):
        # CSI shape = len(threshold)
        csi = self._CSI
        trhd = self._trhd
        print('The CSI score:')
        for i in range(len(trhd)):
            print(f'\n{trhd[i]:2d} => {csi[i]:.3f}')

def idxFromDatetime(time_list, target_t):
    idx=0
    while idx < len(time_list):
        if time_list[idx] == target_t:
            return idx
        idx += 1
    print('your target time is not in the given time range.')

def count(ground, threshold=[1,3,5,10,15,20,30,40]) -> (np.array):
    ratio = np.array([])
    for i in range(len(threshold)):
        if i == len(threshold)-1:
            tmp = np.sum(ground>=threshold[i])
            ratio = np.append(ratio, tmp)
            del tmp
        else:
            tmp = np.sum((ground >= threshold[i])&(ground<threshold[i+1]))
            ratio = np.append(ratio, tmp)
            del tmp
    assert ratio.sum() == np.sum(ground >= threshold[0])
    return ratio.astype(np.int32)
    
class load_data():
    def __init__(self, 
                 *fileNames, 
                 file_num: int=0, 
                 threshold: int=200,
                 ):
        self._files = fileNames
        self._fNum = file_num
        self._threshold = threshold
        self._return = self.eliminate_bad_ass(self._threshold, 
                                              **self._load(self._files),
                                             )
        
    def _load(self, files) -> (dict):
        data_container = []
        for id, file in enumerate(files, start=1):
            # load model output
            if file.endswith('.npz'):
                data = np.load(file)
                data_container.append(data['model_output'])
            elif file.endswith('.pkl'):
                with open(file, 'rb') as f:
                    _, output, _ = pickle.load(f)
                data_container.append(output)
            else:
                raise RuntimeError(f'{file} gets an unknown file format.')
            # laod datetime & target from last file
            if id == self._fNum:
                with open(file, 'rb') as f:
                    datetime, _, target = pickle.load(f)
            print(f'{file} has been loaded.')
        return {'dc': data_container, 'tg': target, 'dt': datetime}
    
    def eliminate_bad_ass(self, thsh, dc, tg, dt):
        assert len(tg.shape) == 4 #[B, 3, H, W]
        error_idx, _, _, _ = np.where(tg > thsh)
        error_idx = sorted(list(set(error_idx)))
        #
        dc = np.stack([*dc], axis=1) # [B, N_models, 3, H, W]
        dc = np.delete(dc, error_idx, axis=0)
        dc = np.split(dc, self._fNum, axis=1)
        #
        tg = np.delete(tg, error_idx, axis=0)
        #
        dt = np.delete(np.array(dt), error_idx, axis=0)
        return dc, tg, list(dt)

class ratio_count():
    def __init__(self, 
                 preds:list, 
                 target, 
                 threshold=[0,1,3,5,10,15,20,30,40,50], 
                 method = count
                 ) -> (list):
        '''
        pred is a list containing np_arrays. Like [3][x, y, z]
        '''
        self._preds = preds
        self._label = target
        self._threshold = threshold
        self._method = method
        self._denominator = self._method(self._label, self._threshold) # size = threshold

    def start(self):
        result = []
        for pred in tqdm(self._preds, ncols=60):
            assert np.shape(pred) == np.shape(self._label), 'Size inconsistency.'
            result.append(self._method(pred, self._threshold) / self._denominator)
        return result

def cal_itv_error(preds:list, target, threshold=[0,5,10,20,30,40]) -> (list) :
    '''
    pred is a list containing np_arrays. Like [3][B, H, W]
    '''
    diff_model = []
    for pred in tqdm(preds, ncols=60):
        assert np.shape(pred) == np.shape(target), 'Size inconsistency.'
        diff_thsh = []
        for i in range(len(threshold)):
            if i == len(threshold)-1:
                x, y, z = np.where(target>=threshold[i])
                diff_thsh.append(pred[x, y, z] - target[x, y, z])
            else:
                x, y, z = np.where((target>=threshold[i]) & (target<threshold[i+1]))
                diff_thsh.append(pred[x, y, z] - target[x, y, z])
        diff_model.append(diff_thsh)
    return diff_model

class error_divided_by_season():
    '''
    calculate the error of summer & winter.
    raw data has the same size as "preds" in cal_itv_error.
    raw_data = [N][B, H, W]
    target = [B, H, W]
    '''
    def __init__(self, raw_data, target, datetime: list, \
                 threshold = [0,5,10,20,30,40], 
                 cal_int_error = cal_itv_error) -> (None):
        self._raw_data = raw_data
        self._target = target
        self._time = datetime
        self._threshold = threshold
        self._method = cal_int_error
    
    def divide_into_seasons(self):
        summer_id, winter_id = self.idx_divide()
        # output data: [N_models][2][B, H, W]
        # output targ: [2][B, H, W]
        return ([[data[summer_id], data[winter_id]] for data in self._raw_data], 
                [self._target[summer_id], self._target[winter_id]]
                )

    def cal_error(self):
        result = []
        data, target = self.divide_into_seasons()
        for n_models in data:
            for season in range(2): # only summer & winter
                result.append(self._method([n_models[season]], 
                                           target[season], 
                                           self._threshold,
                                          )[0]
                             )
        return result

    def idx_divide(self):
        summer = [6, 7, 8]; winter = [12, 1, 2]
        summer_id = []; winter_id = []
        for id, dt in enumerate(self._time):
            if dt.month in summer:
                summer_id.append(id)
            elif dt.month in winter:
                winter_id.append(id)
        return summer_id, winter_id
    
    def time_divide(self):
        summer_id, winter_id = self.idx_divide()
        summer_time = []; winter_time = []
        for i in summer_id:
            summer_time.append(self._time[i])
        for j in winter_id:
            winter_time.append(self._time[j])
        return [summer_time, winter_time]
    
class divide_dayperiod():
    def __init__(self, input_d, datetime):
        # input_d = [B, H, W]
        self._inputd = input_d
        self._dt = datetime
        
    def execute(self):
        container_day = [];container_night = []
        for i, d in enumerate(self._dt):
            if d.hour in range(6, 18):
                container_day.append(self._inputd[i])
            else:
                container_night.append(self._inputd[i])
        return [np.stack(container_day,axis=0), np.stack(container_night,axis=0)]
    
def cal_mape(inp, tar, thsh, is_larger=True, is_less=False):
    if is_larger:
        mask = np.ma.masked_greater_equal(tar, thsh).mask
    elif is_less:
        mask = (np.ma.masked_less(tar, thsh).mask) & (np.ma.masked_greater_equal(tar, 1).mask)
    
    new_data = inp[mask]
    new_targ = tar[mask]
    mape = np.mean(np.abs((new_data - new_targ) / new_targ))
    
    if is_larger:
        # show case numbers
        mask = mask * 1
        mask = np.sum(mask, axis=(-1, -2))
        print('case numbers: ', np.sum(mask != 0))
    return mape