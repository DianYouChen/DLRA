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





