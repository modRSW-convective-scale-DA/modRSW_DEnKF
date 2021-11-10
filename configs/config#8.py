####################################################################
##  FILE CONTAINING CONFIGURATION PARAMENTERS            	  ##
####################################################################
'''
List of fixed parameters for model integration and EnKF.
'''

import numpy as np
from init_cond_modRSW import init_cond_topog_cos

'''Output directory'''
outdir = '/nobackup/mmlca/kent_enkf/2xres_1h/config#8'
IAU_dir = ''

''' MODEL PARAMETERS '''

Neq = 3     # number of equations in system (3 with topography, 4 with rotation_
L = 1.0     # length of domain (non-dim.)

Nk_fc = 200                                 # forecast resolution
dres = 2                                     # refinement factor for truth gridsize
Nk_tr = dres*Nk_fc                           # truth resolution
n_d = Neq * Nk_fc                   # total number of variables (dgs of freedom)

cfl_fc = 0.5 # Courant Friedrichs Lewy number for time stepping
cfl_tr = 0.5

Ro = 'Inf'          # Rossby no. Ro ~ V0/(f*L0)
Fr = 1.1            # froude no.
g = Fr**(-2) 		# effective gravity, determined by scaling.
A = 0.1
V = 1.

# CHOOSE INITIAL CONDITION FROM init_cond_modRSW: 
ic = init_cond_topog_cos 

# threshold heights
H0 = 1.0
Hc = 1.02
Hr = 1.05

# remaining parameters related to hr
beta = 0.2
alpha2 = 10
#c2 = g*Hr
cc2 = 0.1*g*Hr

''' FILTER PARAMETERS '''

n_ens = 18                              # number of ensembles
TIMEOUT_PAR = n_ens*4			# time to wait until all forecasts running in parallel are over
Nmeas = 48                              # number of cycles
Nforec = 13				# duration of each forecast in dtmeasure
NIAU = 1000				# suppress injection of additional inflation with IAU for the first NIAU hours since assimilation 
tn = 0.0                                # initial time
spin_up = 12
#tmax = Nmeas*0.144                      # end time = Nmeas*1hr real time
#dtmeasure = tmax/Nmeas                  # length of each window
dtmeasure = 0.144  
tmax = (Nmeas+Nforec)*dtmeasure
t_end_assim = Nmeas*dtmeasure
tmeasure = dtmeasure
assim_time = np.linspace(tn,t_end_assim,Nmeas+1) # vector of times when system is observed
lead_times = [0,3,6,9,12]
#
sig_ic = [0.1,0.05,0.0]                 # initial ens perturbations [h,hu,hr]
ob_noise = [0.05,0.025,0.002]            # ob noise for [h,u,r]
#model_noise = [0.02, 0.02, 0.0]        # model-error standard deviations
obs_h_d = 25
obs_u_d = 20
obs_r_d = 25
n_obs_h = Nk_fc // obs_h_d
n_obs_u = Nk_fc // obs_u_d
n_obs_r = Nk_fc // obs_r_d
n_obs = n_obs_h + n_obs_u + n_obs_r
#o_d = 20                               # ob density: observe every o_d elements
#n_obs = n_d / o_d                      # no. of observations (divides Nk_fc)
#

''' OUTER LOOP'''
'''
Parameters for outer loop are specified in main_p.py 
loc     : localisation scale
add_inf : additive infaltaion factor
rtpp    : Relaxation to Prior Perturbations scaling factor
rtps    : Relaxation to Prior Spread scaling factor
'''
# MUST BE FLOATING POINT
loc = [ 0.5, 1.0, 1.5, 2.0 ]
add_inf = [ 0.1, 0.15, 0.2, 0.3, 0.4 ]
rtpp = [0.5]
rtps = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

##################################################################
#			END OF PROGRAM				 #
##################################################################
