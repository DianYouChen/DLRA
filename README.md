## The DeepQPF-PONI Model Codebase

This repository presents the contents of the DeepQPF-PONI model, which is trained with rain rate (mm/h), radar reflectivity (dBZ), and heterogeneous weather data. 
The model input incorporates QPESUMS rain rate, column value (CV), and environment variables from ERA5. 
This model predicts accumulated hourly rainfall at a lead time of three hours.

### How to build the environment
``` python=1
# update conda in base env
conda update conda

# create a new env for this project
conda create --name deepQPF-PONI

# turn off the auto-activate
conda config --set auto_activate_base false

# activate env
conda activate deepQPF-PONI

# install packages
conda install -c conda-forge jupyterlab
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip3 install geopandas
conda install -c anaconda netcdf4 -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge cartopy -y
conda install -c conda-forge cmaps -y
pip install scipy
pip install pytorch-lightning # version==1.6.3
pip install pytorch-msssim
pip install test-tube
pip install wget
pip install metpy
pip install tqdm
pip install wandb
pip install python-crontab

pip install cdsapi # for EC climate data
conda install -c anaconda basemap -y
```
### Locally connected network
Since the current version of PyTorch does not include locally connected network algorithms, users need to manually add the `Conv2dLocal` module to the PyTorch source code. Please refer to the following pull request for detailed instructions and code implementation:https://github.com/pytorch/pytorch/pull/1583/files.






