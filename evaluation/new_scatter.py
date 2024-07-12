import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

font = {'family'     : 'sans-serif',
        'weight'     : 'bold',
        'size'       : 14
        }
axes = {'titlesize'  : 16,
        'titleweight': 'bold',
        'labelsize'  : 14,
        'labelweight': 'bold'
        }
mpl.rc('font', **font)  # pass in the font dict as kwargs
mpl.rc('axes', **axes)

class scatterPlot(): 
    def __init__(self, input_d, target, thsh, limit):
        self._input = input_d # [B, H, W]
        self._target = target # [B, H, W]
        self._thsh = thsh
        self.x, self.y = self.filter()
        self._lim = limit

    def filter(self):
        mask = self._target >= self._thsh
        return self._input[mask], self._target[mask]
    
    def correlation(self):
        return np.corrcoef(self.x, self.y)
#         return np.corrcoef(self._input.reshape(-1), self._target.reshape(-1))

    def plot(self):
        fig, ax = plt.subplots(1,1, figsize=(10, 7.5), dpi=200, facecolor='w')
        hbin = ax.hist2d(self.y, self.x, 
                         norm = colors.LogNorm(),
                         bins=100,
                         cmap = "RdYlGn_r")
        ax.set(xlim=(self._lim[0], self._lim[1]), 
               ylim=(self._lim[0], self._lim[1]))
        ax.set_xlabel("ground truth (mm)")
        ax.set_ylabel("prediction (mm)")
        # write corrcoef
        ax.text(0, self._lim[1]*1.01, 
                "correlation coefficient: {0:.4f}".format(self.correlation()[0,1]))
        fig.colorbar(hbin[3], ax=ax)