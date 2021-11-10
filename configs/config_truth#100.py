####################################################################
##            FILE CONTAINING CONFIGURATION PARAMETERS            ##
####################################################################

import numpy as np
from init_cond_modRSW import init_cond_rest, init_cond_4

'''Output directory'''
outdir = '/nobackup/mmlca/test_modRSW/config_truth#100'

# CHOOSE INITIAL PROFILE FROM init_cond_isenRSW
ic = init_cond_rest

# CHOOSE BOUNDARY CONDITIONS
kx1 = 8*np.pi
H_pert = 0.0001

# NUMERICAL PARAMETERS
cfl = 0.1							# Courant Friedrichs Lewy number for time stepping
Neq = 4                                                         # number of equations
L = 1. 								# lenght of domain (non-dim)
Nk = 800							# truth resolution

# DYNAMICAL PARAMETERS
Fr = 0.37971 
Ro = 0.5                                                        # Rossby number
g = 1./Fr**2                                                    # gravity acceleration
H0 = 1.0							# pseudo-density initial state
A = 0.05							
V = 1.								

# threshold (for pseudo-density)
Hc = 2.
Hr = 2.1

# Rain and convection parameters
alpha2 = 20.
beta = 0.4
cc2 = 5.

''' FORECAST PARAMETERS '''
tn = 0.0                                			# initial time
Nmeas = 48
hour = 0.072
tmax = Nmeas*hour
dtmeasure = hour
tmeasure = dtmeasure

