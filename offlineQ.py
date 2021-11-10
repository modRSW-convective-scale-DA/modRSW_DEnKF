#######################################################################
# Investigating computation and structure of model error and candidate Q matrices
#######################################################################
import numpy as np
import importlib
import sys
import matplotlib.pyplot as plt
#from parameters import *
from scipy import linalg
from f_modRSW import make_grid, step_forward_topog, time_step, step_forward_modRSW

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################

spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
Nk_fc = config.Nk_fc
L = config.L
V = config.V
Nmeas = config.Nmeas
Neq = config.Neq
dres = config.dres
cfl_fc = config.cfl_fc
Hc = config.Hc
Hr = config.Hr
Ro = config.Ro
assim_time = config.assim_time
dtmeasure = config.dtmeasure
cc2 = config.cc2
beta = config.beta
alpha2 = config.alpha2
g = config.g
#model_noise = config.model_noise
U_relax = config.U_relax
tau_rel = config.tau_rel

# Q-matrix from pre-defined model noise
#var_h = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[0]
#var_u = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[1]
#var_r = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[2]
#Q = np.diag(np.concatenate((var_h,var_u,var_r)))


##################################################################
# Cycle over truth trajectory times.  Use each value as the initial condition
# for both the truth (obtained from X_tr at the next timestep) and a
# lower-resolution forecast.
B = np.load(str(outdir + '/B_tr.npy')) # truth
U_tr = np.load(str(outdir + '/U_tr_array_2xres_1h.npy')) # truth
fc_grid = make_grid(Nk_fc, L) # forecast
Kk_fc = fc_grid[0]
xc = fc_grid[2]
U_rel = U_relax(Neq,Nk_fc,L,V,xc,U_tr[:,0::dres,0])
nsteps = Nmeas
X = np.zeros([Neq * Nk_fc, nsteps])
U = np.copy(U_tr[:, 0::dres, :]) # Sample truth at forecast resolution
for T in range(nsteps):
    tn = assim_time[T]
    print(tn)
    tmeasure = tn+1*dtmeasure
    print(tmeasure)
    U_fc = np.copy(U[:, :, T])
    while tn < tmeasure:
        dt = time_step(U_fc, Kk_fc, cfl_fc, cc2, beta, g) # compute stable time step
        tn = tn + dt
        if tn > tmeasure:
            dt = dt - (tn - tmeasure) + 1e-12
            tn = tmeasure + 1e-12
        U_fc = step_forward_modRSW(U_fc, U_rel, dt, Neq, Nk_fc, Kk_fc, Ro, alpha2, Hc, Hr, cc2, beta, g, tau_rel)
#        U_fc = step_forward_topog(U_fc, B, dt, tn, Neq, Nk_fc, Kk_fc, Hc, Hr, cc2, beta, alpha2, g)
    X[:, T] = U[:, :, T+1].flatten() - U_fc.flatten()
print("Means of proxy error components = ", \
    np.mean(U[:, :, 1:(nsteps+1)]
            - np.repeat(U_fc, nsteps).reshape(Neq, Nk_fc, nsteps), axis=(1, 2)))

# Compute the covariance matrix.
Q = np.cov(X, bias=False)

#Pose the covariance of r to 0
Q_diag = np.array(Q.diagonal())
Q_diag[3*Nk_fc:] = 0.0

#Pose the covariance of u to 0
#Q_diag = np.array(Q.diagonal())
#Q_diag[Nk_fc:2*Nk_fc] = 0.0

#Return a diagonal matrix
Q = np.diag(Q_diag)
#Q = np.diag(Q.diagonal())

# Recondition Q to have a maximum condition number of kappa following
# Smith et al.: doi:10.1002/2017GL075534, modified to use singular value
# decomposition.
#u, s, vh = linalg.svd(Q)
#print "Raw Q has 2-norm condition number = ", s[0] / s[-1]

# Apply the required eigenvalue offset and reconstruct Q using U alone
# to ensure positive semi-definiteness.
#kappa = 1000
#lam = (s[0] - kappa * s[-1]) / (kappa - 1)
#print "Reconditioning singular value offset = ", lam
#s += lam
#Q = np.dot(np.dot(u, np.diag(s)), np.transpose(u))
#print "Reconditioned Q has 2-norm condition number = ", s[0] / s[-1]

# Check for positive semi-definiteness.
#try:
#    tmp = np.linalg.cholesky(Q)
#except:
#    raise

#Q = np.diag(np.sum(X**2, axis=1) / nsteps)

print(("Min. and max. variances", np.amin(Q.diagonal()), np.amax(Q.diagonal())))
np.save(str(outdir + '/Qnew'), Q)

# Plot the Q matrix.
#fig = plt.figure(1)
#ax = fig.add_subplot(111)
#cax = ax.pcolormesh(Q, vmin=-0.02, vmax=0.02, cmap='RdBu_r')
#fig.colorbar(cax)
#fig.show()

# Correlation matrix.
#Qcorr = np.corrcoef(X, bias=False)
#fig = plt.figure(2)
#ax = fig.add_subplot(111)
#cax = ax.pcolormesh(Qcorr, vmin=-1, vmax=1, cmap='RdBu_r')
#fig.colorbar(cax)
#fig.show()

#raw_input() # Press 'Enter' to close
