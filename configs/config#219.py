####################################################################
##            FILE CONTAINING CONFIGURATION PARAMETERS            ##
####################################################################

import numpy as np
from init_cond_modRSW import init_cond_9
from relax_sol_modRSW import U_relax_10

'''Output directory'''
outdir = '/nobackup/mmlca/DA_modRSW/config#219'

# CHOOSE INITIAL PROFILE FROM init_cond_isenRSW
ic = init_cond_9

# NUMERICAL PARAMETERS
cfl_fc = 0.1							# Courant Friedrichs Lewy number for time stepping
cfl_tr = 0.1
Neq = 4                                                         # number of equations
L = 1. 								# lenght of domain (non-dim)
Nk_fc = 200							# forecast resolution
dres = 2
Nk_tr = Nk_fc*dres
n_d = Neq*Nk_fc

R = 287.
theta1 = 300.
theta2 = 290.

# DYNAMICAL PARAMETERS
Fr = 20./np.sqrt((10./theta1)*R*theta2*0.15)
Ro = 0.4                                                        # Rossby number
g = 1./Fr**2                                                    # gravity acceleration
H0 = 1.0							# pseudo-density initial state
A = 0.2/0.15 - H0							
V = 1.							

# threshold (for pseudo-density)
Hc = 0.16/0.15
Hr = 0.2/0.15

# Rain and convection parameters
alpha2 = 5.
beta = 0.8
cc2 = 2.

# Relaxation solution
U_relax = U_relax_10
tau_rel = 4.

''' FILTER PARAMETERS '''

n_ens = 25                              # number of ensembles
TIMEOUT_PAR = n_ens*8                   # time to wait until all forecasts running in parallel are over
Nmeas = 48                              # number of cycles
Nforec = 7                             # duration of each forecast in dtmeasure
NIAU = 1000                            # suppress injection of additional inflation with IAU for the first NIAU hours since assimilation 
tn = 0.0                                # initial time
spin_up = 12
dtmeasure = 0.144
tmax = (Nmeas+Nforec)*dtmeasure
t_end_assim = Nmeas*dtmeasure
tmeasure = dtmeasure
assim_time = np.linspace(tn,t_end_assim,Nmeas+1) # vector of times when system is observed
lead_times = [0,3,6]
sig_ic = [0.1,0.05, 0.05, 0.0]                 # initial ens perturbations [h,hu,hr]
ob_noise = [0.05,0.02,0.02,0.003]            # ob noise for [h,u,r]
#model_noise = [0.02, 0.02, 0.0]        # model-error standard deviations
obs_h_d = 25
obs_u_d = 20
obs_v_d = 20
obs_r_d = 20
n_obs_h = Nk_fc // obs_h_d
n_obs_u = Nk_fc // obs_u_d
n_obs_v = Nk_fc // obs_v_d
n_obs_r = Nk_fc // obs_r_d
n_obs = n_obs_h + n_obs_u + n_obs_v + n_obs_r

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
add_inf = [ 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
rtpp = [0.5]
rtps = [ 0.1, 0.3, 0.5, 0.7, 0.9 ]

##################################################################
#                       END OF PROGRAM                           #
##################################################################

