# modRSW_DEnKF
## Idealised DA experiments with the DEnKF and modRSW model and their relevance for convective-scale NWP

This repository contains the relevant source code and documentation for the research paper: 

```Kent et al (2020): Idealized forecast-assimilation experiments for convective-scale Numerical Weather Prediction;```

currently published on [EarthArxiv](https://eartharxiv.org/repository/view/1921/). It suceeds the preliminary forecast-assimilation system developed during TK's PhD (Kent 2016) which uses an idealised fluid model of convective-scale Numerical Weather Prediction (modRSW; Kent et al. 2017) and the perturbed-observation Ensemble Kalman Filter (EnKF). This README contains sufficient instruction for users to download, implement and adapt the source code, which briefly comprises Python3 scripts for the numerical solver for the discretised model, data assimilation algorithms, plotting and data analysis. 




## Background
...

## References
* Kent, T. (2016): An idealised fluid model of Numerical Weather Prediction: dynamics and data assimilation. *PhD thesis, University of Leeds*. Available at [http://etheses.whiterose.ac.uk/17269/](http://etheses.whiterose.ac.uk/17269/).

* Kent, T., Bokhove, O., Tobias, S.M. (2017): Dynamics of an idealised fluid model for investigating convective-scale data assimilation. *Tellus A: Dynamic Meteorology and Oceanography*, **69(1)**, 1369332. [DOI](https://www.tandfonline.com/doi/full/10.1080/16000870.2017.1369332).

* Kent, T., Cantarello, L., Inverarity, G.W., Tobias, S.M., Bokhove, O. (2020): Idealised forecast-assimilation experiments for convective-scale Numerical Weather Prediction. *EarthArXiv*, [DOI](https://eartharxiv.org/repository/view/1921/). 
----

## Getting started
### Versions -- Check!!
All of the source code is written in Python and relies heavily on the numpy module, amongst others. The plotting routines require matplotlib. The versions used in this development are tabled below. Other versions may work of course, but have not been tested by the authors.

Software      | Version
------------- | -------------
Python  | 3.8.8
Matplotlib  | 3.4.3
Numpy  | 1.21.2

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
* click on the download link on the repository homepage [https://github.com/modRSW-convective-scale-DA/modRSW_DEnKF](https://github.com/modRSW-convective-scale-DA/modRSW_DEnKF) and save to desired directory.

### Running the code: basics
...

To kill at any point, press ```Ctrl+c```, or kill the active processes using ```top``` from the terminal.


## Brief overview of files

...

### Model only

...

### Assimilation framework

...

### Plotting and data analysis

...

### .npy data

...

## Test cases


