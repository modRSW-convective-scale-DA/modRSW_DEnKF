#######################################################################
# Perturbed obs. EnKF for 1.5D SWEs with rain variable and topography
#######################################################################

'''
May 2017
Main run script for batch-processing of EnKF jobs. 
Define outer-loop through parameter space and run the EnKF subroutine for each case.
Summary:
> truth generated outside of outer loop as this is the same for all experiments 
> uses subroutine <subr_enkf_modRSW_p> that parallelises ensemble forecasts using multiprocessing module
> Data saved to automatically-generated directories and subdirectories with accompanying readme.txt file.
'''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno
import sys
import importlib.util

# HANDLE WARNINGS AS ERRORS
##################################################################
import warnings
warnings.filterwarnings("error")

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

#from parameters import *
#from f_modRSW import make_grid 
#from f_enkf_modRSW import generate_truth
#from init_cond_modRSW import init_cond_topog4, init_cond_topog_cos
#from create_readme import create_readme
from subr_enkf_modRSW_p import run_enkf

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
ic = config.ic
add_inf = config.add_inf
rtpp = config.rtpp
rtps = config.rtps 

### LOAD TRUTH, OBSERVATIONS AND OBSERVATION OPERATOR ###

f_path_name = str(outdir+'/U_tr_array_2xres_1h.npy')
f_obs_name = str(outdir+'/Y_obs_2xres_1h.npy')
f_H_name = str(outdir+'/H.npy')

try:
    ' *** Loading truth trajectory... *** '
    U_tr_array = np.load(f_path_name)
except:
    print(' Failed to find the truth trajectory: run create_truth+obs.py first')

try:
    H = np.load(f_H_name)
except:
    print('Failed to load the observation operator H:  run create_truth+obs.py first')

try:
    Y_obs = np.load(f_obs_name)
except:
    print('Failed to load the observations: run create_truth+obs.py first')

##################################################################    
# EnKF: outer loop 
##################################################################
print(' ')
print(' ------- ENTERING EnKF OUTER LOOP ------- ')  
print(' ')
#for i in range(0,len(loc)):
i = int(sys.argv[2])-1
for j in range(0,len(add_inf)):
    for k in range(0,len(rtpp)):
        for l in range(0,len(rtps)):
            run_enkf(i, j, k, l, ic, U_tr_array, Y_obs, H, outdir, sys.argv[1])
print(' ')   
print(' --------- FINISHED OUTER LOOP -------- ')
print(' ')   
print(' ------------ END OF PROGRAM ---------- ')  
print(' ') 
    
##################################################################    
#                        END OF PROGRAM                          #
##################################################################
