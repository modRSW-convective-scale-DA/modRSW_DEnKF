####################################################################
##            FILE CONTAINING CONFIGURATION PARAMETERS            ##
####################################################################

import numpy as np
from init_cond_modRSW import init_cond_topog_cos
from relax_sol_modRSW import U_relax_0

'''Output directory'''
outdir = '/nobackup/mmlca/test_modRSW/config_truth#64'

# CHOOSE INITIAL PROFILE FROM init_cond_isenRSW
ic = init_cond_topog_cos

# NUMERICAL PARAMETERS
cfl = 0.5							# Courant Friedrichs Lewy number for time stepping
Neq = 3                                                         # number of equations
L = 1. 								# lenght of domain (non-dim)
Nk = 400							# truth resolution

# DYNAMICAL PARAMETERS
Fr = 1.1
Ro = float('inf')                                                      # Rossby number
g = 1./Fr**2                                                    # gravity acceleration
H0 = 1.0							# pseudo-density initial state
A = 0.1							
V = 1.							

# threshold (for pseudo-density)
Hc = 1.02
Hr = 1.05

# Rain and convection parameters
alpha2 = 10.
beta = 0.2
cc2 = 0.085

# Relaxation solution
U_relax = U_relax_0
tau_rel = float('inf')

''' FORECAST PARAMETERS '''
tn = 0.0                                			# initial time
Nmeas = 48
hour = 0.144
tmax = Nmeas*hour
dtmeasure = hour
tmeasure = dtmeasure

