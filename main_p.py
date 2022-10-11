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
import shutil

# HANDLE WARNINGS AS ERRORS
##################################################################
import warnings
warnings.filterwarnings("error")

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

from subr_enkf_modRSW_p import run_enkf

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
config_string = sys.argv[1]
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
ic = config.ic
add_inf = config.add_inf
loc = config.loc
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
for i in range(0,len(loc)):
    for j in range(0,len(add_inf)):
        for k in range(0,len(rtpp)):
            for l in range(0,len(rtps)):
                print(i,j,k,l)
                if __name__ == '__main__': 
                    run_enkf(i, j, k, l, ic, U_tr_array, Y_obs, H, outdir, config_string)
print(' ')   
print(' --------- FINISHED OUTER LOOP -------- ')
print(' ')   
print(' ------------ END OF PROGRAM ---------- ')  
print(' ') 
    
##################################################################    
#                        END OF PROGRAM                          #
##################################################################
