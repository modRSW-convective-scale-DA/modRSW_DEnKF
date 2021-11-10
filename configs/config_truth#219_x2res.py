####################################################################
##            FILE CONTAINING CONFIGURATION PARAMETERS            ##
####################################################################

import numpy as np
from init_cond_modRSW import init_cond_9
from relax_sol_modRSW import U_relax_10

'''Output directory'''
outdir = '/nobackup/mmlca/test_modRSW/config_truth#219_x2res'

# CHOOSE INITIAL PROFILE FROM init_cond_isenRSW
ic = init_cond_9

# NUMERICAL PARAMETERS
cfl = 0.1							# Courant Friedrichs Lewy number for time stepping
Neq = 4                                                         # number of equations
L = 1. 								# lenght of domain (non-dim)
Nk = 800							# truth resolution

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

''' FORECAST PARAMETERS '''
tn = 0.0                                			# initial time
Nmeas = 96
hour = 0.072
tmax = Nmeas*hour
dtmeasure = hour
tmeasure = dtmeasure

