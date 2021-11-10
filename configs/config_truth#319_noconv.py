####################################################################
##            FILE CONTAINING CONFIGURATION PARAMETERS            ##
####################################################################

import numpy as np
from init_cond_modRSW import init_cond_9
from relax_sol_modRSW import U_relax_12

'''Output directory'''
outdir = '/nobackup/mmlca/test_modRSW/config_truth#319_noconv'

# CHOOSE INITIAL PROFILE FROM init_cond_isenRSW
ic = init_cond_9

# NUMERICAL PARAMETERS
cfl = 0.1							# Courant Friedrichs Lewy number for time stepping
Neq = 4                                                         # number of equations
L = 1. 								# lenght of domain (non-dim)
Nk = 400							# truth resolution

R = 287.
theta1 = 311.
theta2 = 291.8

# DYNAMICAL PARAMETERS
Fr = 12.4/np.sqrt((19.2/theta1)*R*theta2*0.2)
Ro = 0.248                                                        # Rossby number
g = 1./Fr**2                                                    # gravity acceleration
H0 = 1.0							# pseudo-density initial state
A = 0.24/0.2 - H0							
V = 1.							

# threshold (for pseudo-density)
Hc = 1./0.2
Hr = 1./0.2

# Rain and convection parameters
alpha2 = 6.
beta = 2.
cc2 = 1.8

# Relaxation solution
U_relax = U_relax_12
tau_rel = 4.

''' FORECAST PARAMETERS '''
tn = 0.0                                			# initial time
Nmeas = 96
hour = 0.089
tmax = Nmeas*hour
dtmeasure = hour
tmeasure = dtmeasure

