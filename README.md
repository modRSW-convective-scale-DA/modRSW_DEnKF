# modRSW_DEnKF
## Idealised data assimilation (DA) experiments with the modRSW model using a Deterministic Kalman Filter (DEnKF) and their relevance for convective-scale Numerical Weather Prediction (NWP) models

This repository contains the relevant source code and documentation for the research paper: 

```Kent et al (2020): Idealized forecast-assimilation experiments for convective-scale Numerical Weather Prediction;```

currently published on [EarthArxiv](https://eartharxiv.org/repository/view/1921/). A slightly revised version has also been submitted to Geoscientific Model Developments for peer-review publication. They suceed the preliminary forecast-assimilation system developed during TK's PhD (Kent 2016) which uses an idealised fluid model of convective-scale Numerical Weather Prediction (modRSW; Kent et al. 2017) and the perturbed-observation Ensemble Kalman Filter (EnKF). This README contains sufficient instruction for users to download, implement and adapt the source code, which briefly comprises Python3 scripts for the numerical solver for the discretised model, data assimilation algorithms, plotting and data analysis. 


For any questions or code bugs please send an email to mmlca@leeds.ac.uk.

## References
#### Thesis + articles:

* Kent, T. (2016): An idealised fluid model of Numerical Weather Prediction: dynamics and data assimilation. *PhD thesis, University of Leeds*. Available at [http://etheses.whiterose.ac.uk/17269/](http://etheses.whiterose.ac.uk/17269/).

* Kent, T., Bokhove, O., Tobias, S.M. (2017): Dynamics of an idealised fluid model for investigating convective-scale data assimilation. *Tellus A: Dynamic Meteorology and Oceanography*, **69(1)**, 1369332. [DOI](https://www.tandfonline.com/doi/full/10.1080/16000870.2017.1369332).

* Kent, T., Cantarello, L., Inverarity, G.W., Tobias, S.M., Bokhove, O. (2020): Idealised forecast-assimilation experiments for convective-scale Numerical Weather Prediction. *EarthArXiv*, [DOI](https://eartharxiv.org/repository/view/1921/). 

#### Presentations:

* Kent, T.: 'The modRSW model -- physical basis, numerics, and dynamics', [DA workshop](https://tkent198.github.io/workshop.html), University of Leeds, 16 May 2019. 
* Kent, T., Inverarity, G., Cantarello, L., Tobias, S.M., Bokhove, O.: 'Idealised forecast-assimilation experiments and their relevance for convective-scale Numerical Weather Prediction', EGU General Assembly, Vienna, 7-12 April 2019. [Slides](https://tkent198.github.io/files/egu2019_DA.pdf).
----

## Getting started
### Versions -- Check!!
All of the source code is written in Python and relies heavily on the numpy module, amongst others. The plotting routines require matplotlib. The versions used in this development are tabled below. Other versions may work of course, but have not been tested by the authors.

Software      | Version
------------- | -------------
Python  | 3.10.4
Matplotlib  | 3.5.3
Numpy  | 1.22.2

To check python version, from the terminal:
```
python --version
```

To check numpy version, open python in the terminal, import it and use the version attribute:
```
>>> import numpy
>>> numpy.__version__
```
Same for all other modules. 

### Download and install

Clone from terminal (recommended):
* Go to the directory where you want to save the repository and use the command:
```
git clone https://github.com/modRSW-convective-scale-DA/modRSW_DEnKF.git
```
* Once downloaded, to get any updates/upgrades to the original clone, use the command:
```
git pull https://github.com/modRSW-convective-scale-DA/modRSW_DEnKF.git
```

Direct download: 
* click on the download link on the repository url [https://github.com/modRSW-convective-scale-DA/modRSW_DEnKF](https://github.com/modRSW-convective-scale-DA/modRSW_DEnKF) and save to desired directory.

### Running the code: basics

All python scripts of the repository can be executed from terminal, from inside the modRSW_DEnKF repository:
```
python name_of_script.py config/config_file.py [additional arguments]
```
To kill at any point, press ```Ctrl+c```, or kill the active processes using ```top``` from the terminal.

### Working out the experiment index

In order to be executed, many of the scripts contained in this repository require as argument an experiment index (exp_idx) associated with the desired set of filter parameters (loc, add_inf, rtps). This index is an integer which depends on the total number of combinations given by the list of parameters defined in the configuration file. For example, given the following set of parameters in the configuration file:
```
loc = [0.5, 1.0]
add_inf = [0.1, 0.2]
rtps = [0.3, 0.6]
```
the total number of parameter combinations would be 8 (2 localisation values x 2 additive inflation values x 2 rtps values). The experiment index exp_idx is therefore defined as an integer between 1 and 8 that loops over each parameter in the following order:
```
exp_idx=1
for i in loc:
  for j in add_inf:
    for k in rtps:
      exp_idx=+1
```

## Brief overview of files
Here is an overview of the files contained in this repository and what they do. They are listed in the order they need to be modified or run.

### Configuration file and look-up table
* *configs/config_example.py*: this file contains all the parameters, file paths and values used for running the modRSW model, creating the observing system and  setting up the data assimilation algorithm.

### Model only
* *run_modRSW.py*: this script is used to run the modRSW model without any data assimilation. It takes the configuration file as only argument:
```
python3 run_modRSW.py configs/config_example.py
```
* *hovmoller.py*: this script can be used to plot a Hovmöller plot of the output of run_modRSW.py, which needs to be run first. It takes the configuration file as only argument:
```
python3 hovmoller.py configs/config_example.py
```

### Assimilation framework
* *create_truth+obs.py*: this script generates the nature run trajectory of the modRSW model and generate the observations. It takes the configuration file as only argument:
```
python3 create_truth+obs.py configs/config_example.py
```
* *offlineQ.py*: this script calculates the model error covariance matrix Q as specified in the configuration file. It needs the script *create_truth+obs.py* to be run first. It takes the configuration file as only argument:
```
python3 offlineQ.py configs/config_example.py
```
* *main_p.py*: this script launches the main data assimilation routine. It needs both the scripts *create_truth+obs.py* and *offlineQ.py* to be run first. It takes the configuration file as only argument:
```
python3 main_p.py configs/config_example.py
```

### Plotting and data analysis
* *plot_func_t.py*: this script generates time series of various domain-average statistics, such as Root Mean Square Error (RMSE), Continuous Ranked Probability Score (CRPS) and Obsevation Influence Diagnostics (OID). It takes four arguments: 1) the configuration file, 2) the experiment index, and 3)-4) two different lead times for the plotting.
```
python3 plot_func_t.py configs/config_example.py exp_index lead_time1 lead_time2
```
* *plot_func_x.py*: this script generates a snapshot of the model solution, the analysis and the observations at the specified analysis time. It takes three argument: 1) the configuration file, 2) the experiment index, and 3) the analysis time.
```
python3 plot_func_x.py configs/config_example.py exp_index analysis_time
```
* *plot_forec_x.py*: this script generates a snapshot of the model solution at the specified validity time (i.e. analysis time plus lead time). It takes three argument: 1) the configuration file, 2) the experiment index, 3) the analysis time, and 4) the lead time.
```
python3 plot_forec_x.py configs/config_example.py exp_index analysis_time lead_time
```
* *compare_stats.py*: this script generates a graphical summary of various diagnostics for all the experiments listed in the configuration file. It takes the configuration file as argument.
```
python3 compare_stats.py configs/config_example.py
```
* *run_modRSW_EFS.py*: this script launches an ensemble of forecast simulations initialised with the desired analysis ensemble and for the selected experiment. The duration of the forecast is specified inside the script by Tfc, which is currently set to ```Tfc=36```. It takes three arguments: 1) the configuration file, 2) the experiment index, and 3) the analysis time.
```
python3 run_modRSW_EFS.py configs/config_example.py exp_index analysis_time
```
* *EFS_stats.py*: this script computes the error doubling times statistics using the output of *run_modRSW_EFS.py*, which needs to be run first (for the same analysis time). It takes three arguments: 1) the configuration time, 2) the experiment index, and 3) the analysis time.
```
python3 EFS_stats.py configs/config_example.py exp_index analysis_time
```
* *err_doub_hist.py*: this script generates the error doubling times histograms with both 1 and 2 hours bins. It takes two arguments: 1) the configuration file, and 2) the experiment index.
```
python3 err_doub_hist.py configs/config_example.py exp_index
```
