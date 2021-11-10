#######################################################################
# FUNCTIONS REQUIRED FOR NUMERICAL INTEGRATION OF THE modRSW MODEL WITH and WITHOUT TOPOGRAPHY
#######################################################################

'''
Module contains numerous functions for the numerical inegration of the modRSW model:
> make_grid            : makes mesh for given length and gridcell number
> NCPflux4d             : calculates numerical flux as per the theory of Kent et al. 2017
> NCPflux_topog         : calcualtes flux for flow over topography
> time_step            : calculates stable time step for integration
> step_forward_modRSW   : integrates forward one time step (forward euler) using NCPflux4d 
> heaviside             : vector-aware implementation of heaviside (also works for scalars).
> ens_forecast_topog    : for use in parallel ensemble forecasting
'''
    
import math as m
import numpy as np
#from parameters import *

##################################################################
#'''-------------- Create mesh at given resolution --------------'''
##################################################################  

# domain [0,L]
def make_grid(Nk,L):
    Kk = L/Nk                   # length of cell
    x = np.linspace(0, L, Nk+1)         # edge coordinates
    xc = np.linspace(Kk/2,L-Kk/2,Nk) # cell center coordinates
    grid = [Kk, x, xc]
    return grid

# domain [-L/2,L/2]:
def make_grid_2(Nk,L):
    Kk = L/Nk                   # length of cell
    x = np.linspace(-0.5*L, 0.5*L, Nk+1)        # edge coordinates
    xc = x - 0.5*Kk # cell center coordinates
    xc = xc[1:]
    grid = [Kk, x, xc]
    return grid

##################################################################
#'''----------------- NCP flux function -----------------'''
##################################################################        
#def NCPflux4d(UL,UR,Hr,Hc,c2,beta,g):
def NCPflux4d(hL,hR,uL,uR,vL,vR,rL,rR,Hr,Hc,c2,beta,g):
### INPUT ARGS:
# UL: left state 4-vector       e.g.: UL = np.array([1,2,0,1]) 
# UR: right state 4-vector      e.g.: UR = np.array([1.1,1.5,0,0.9])
# Hr: threshold height r        e.g.: Hr = 1.15 (note that Hc < Hr)
# Hc: threshold height c        e.g.: Hc = 1.02
# c2: constant for rain geop.   e.g.: c2 = 9.8*Hr/400
# beta: constant for hr eq.     e.g.: beta = 1
# g: scaled gravity             e.g.: g = 1/(Fr**2) 

### OUTPUT:
# Flux: 4-vector of flux values between left and right states
# VNC : 4-vector of VNC values due to NCPs

    # isolate variables for flux calculation    
    #hL = UL[0] 
    #hR = UR[0]
    #uL = UL[1]/hL;
    #uR = UR[1]/hR;
    #vL = UL[2]/hL;
    #vR = UR[2]/hR;
    #rL = UL[3]/hL;
    #rR = UR[3]/hR;

    UL = np.array([hL, hL*uL, hL*vL, hL*rL])
    UR = np.array([hR, hR*uR, hR*vR, hR*rR])

    #    '''wave speeds'''
    SL = min(uL - m.sqrt(c2*beta*np.heaviside(uL-uR,1.)*np.heaviside(hL-Hr,1.) + g*hL*np.heaviside(Hc-hL,1.)),uR - m.sqrt(c2*beta*np.heaviside(uL-uR,1.)*np.heaviside(hR-Hr,1.) + g*hR*np.heaviside(Hc-hR,1)))
    SR = max(uL + m.sqrt(c2*beta*np.heaviside(uL-uR,1.)*np.heaviside(hL-Hr,1.) + g*hL*np.heaviside(Hc-hL,1.)),uR + m.sqrt(c2*beta*np.heaviside(uL-uR,1.)*np.heaviside(hR-Hr,1.) + g*hR*np.heaviside(Hc-hR,1.)))


    #    '''%%%----- For calculating NCP components... -----%%%'''
    a = hR-hL
    b = hL-Hr

    #    '''%%%----- compute the integrals as per the theory -----%%%'''
    if a == 0:
        Ibeta = beta*np.heaviside(uL-uR,1.)*np.heaviside(b,1.)
        Itaubeta = 0.5*beta*np.heaviside(uL-uR,1.)*np.heaviside(b,1.)
    else:    
        d = (a+b)/a
        e = (np.square(a) - np.square(b))/np.square(a)
        Ibeta = beta*np.heaviside(uL-uR,1.)*(d*np.heaviside(a+b,1.) - (b/a)*np.heaviside(b,1.))
        Itaubeta = 0.5*beta*np.heaviside(uL-uR,1.)*(e*np.heaviside(a+b,1.) + (np.square(b)/np.square(a))*np.heaviside(b,1.))
    

    #    '''%%%----- VNC component -----%%%'''
    VNC1 = 0.
    VNC2 = -c2*(rL-rR)*0.5*(hL+hR)
    VNC3 = 0. 
    VNC4 = -np.heaviside(uL-uR,1.)*(uL-uR)*(hR*Ibeta - (hL-hR)*Itaubeta)

    VNC = np.array([VNC1, VNC2, VNC3, VNC4])
    
    #   '''%%%----- Vector flux P^NC in the theory -----%%%'''
    if SL >= 0:
        PhL = 0.5*g*(np.square(hL) + (np.square(Hc) - np.square(hL))*np.heaviside(hL-Hc,1.))
        FluxL = np.array([hL*uL, hL*np.square(uL) + PhL, hL*uL*vL, hL*uL*rL])
        Flux = FluxL - 0.5*VNC
    elif SR <= 0:
        PhR = 0.5*g*(np.square(hR) + (np.square(Hc) - np.square(hR))*np.heaviside(hR-Hc,1.))
        FluxR = np.array([hR*uR, hR*np.square(uR) + PhR, hR*uR*vR, hR*uR*rR])
        Flux = FluxR + 0.5*VNC
    elif SL < 0 and SR > 0:
        PhL = 0.5*g*(np.square(hL) + (np.square(Hc) - np.square(hL))*np.heaviside(hL-Hc,1.))
        PhR = 0.5*g*(np.square(hR) + (np.square(Hc) - np.square(hR))*np.heaviside(hR-Hc,1.))
        FluxR = np.array([hR*uR, hR*np.square(uR) + PhR, hR*uR*vR, hR*uR*rR])
        FluxL = np.array([hL*uL, hL*np.square(uL) + PhL, hL*uL*vL, hL*uL*rL])
        FluxHLL = (FluxL*SR - FluxR*SL + SL*SR*(UR - UL))/(SR-SL)
        Flux = FluxHLL - (0.5*(SL+SR)/(SR-SL))*VNC

    return Flux, VNC

##################################################################
#'''--------------- NCP flux function with topog -------------'''#
##################################################################

def NCPflux_topog(UL,UR,BL,BR,Hr,Hc,c2,beta,g):

    ### INPUT ARGS:
    # UL: left state 3-vector       e.g.: UL = np.array([1,2,0])
    # UR: right state 3-vector      e.g.: UR = np.array([1.1,1.5,0])
    # BL: left B value
    # BR: right B balue
    # Hr: threshold height r        e.g.: Hr = 1.15 (note that Hc < Hr)
    # Hc: threshold height c        e.g.: Hc = 1.02
    # c2: constant for rain geop.   e.g.: c2 = 9.8*Hr/400
    # beta: constant for hr eq.     e.g.: beta = 1
    # g: scaled gravity             e.g.: g = 1/(Fr**2)

    ### OUTPUT:
    # Flux: 4-vector of flux values between left and right states
    # VNC : 4-vector of VNC values due to NCPs

    # isolate variables for flux calculation
    hL = UL[0]
    hR = UR[0]
    
    if hL < 1e-9:
        uL = 0
        rL = 0
    else:
        uL = UL[1]/hL
        rL = UL[2]/hL

    if hR < 1e-9:
        uR = 0
        rR = 0
    else:
        uR = UR[1]/hR
        rR = UR[2]/hR


    zL = hL + BL;
    zR = hR + BR;


    # compute left and right wave speeds (eq. 17)
    SL = min(uL - m.sqrt(c2*beta*heaviside(uL-uR)*heaviside(zL-Hr) + g*hL*heaviside(Hc-zL)),uR - m.sqrt(c2*beta*heaviside(uL-uR)*heaviside(zR-Hr) + g*hR*heaviside(Hc-zR)))
    SR = max(uL + m.sqrt(c2*beta*heaviside(uL-uR)*heaviside(zL-Hr) + g*hL*heaviside(Hc-zL)),uR + m.sqrt(c2*beta*heaviside(uL-uR)*heaviside(zR-Hr) + g*hR*heaviside(Hc-zR)))


    # For calculating NCP components
    a = zR-zL
    b = zL-Hr

    # compute the integrals as per the theory
    if a==0:
        Ibeta = beta*heaviside(uL-uR)*heaviside(b)
        Itaubeta = 0.5*beta*heaviside(uL-uR)*heaviside(b)
    else:
        d = (a+b)/a
        e = (a**2 + b**2)/a**2
        ee = (a**2 - b**2)/a**2
        Ibeta = beta*heaviside(uL-uR)*(d*heaviside(a+b) - (b/a)*heaviside(b))
        Itaubeta = 0.5*beta*heaviside(uL-uR)*(ee*heaviside(a+b) + (b**2/a**2)*heaviside(b))

    VNC1 = 0
    VNC2 = -c2*(rL-rR)*0.5*(hL+hR);
    VNC3 = -heaviside(uL-uR)*(uL-uR)*(hR*Ibeta - (hL-hR)*Itaubeta)

    VNC = np.array([VNC1, VNC2, VNC3])

    if SL > 0:
        PhL = 0.5*g*(hL**2 + ((Hc-BL)**2 - hL**2)*heaviside(zL-Hc))
        FluxL = np.array([hL*uL, hL*uL**2 + PhL, hL*uL*rL])
        Flux = FluxL - 0.5*VNC
    elif SR < 0:
        PhR = 0.5*g*(hR**2 + ((Hc-BR)**2 - hR**2)*heaviside(zR-Hc));
        FluxR = np.array([hR*uR, hR*uR**2 + PhR, hR*uR*rR])
        Flux = FluxR + 0.5*VNC
    elif SL < 0 and SR > 0:
        PhL = 0.5*g*(hL**2 + ((Hc-BL)**2 - hL**2)*heaviside(zL-Hc))
        PhR = 0.5*g*(hR**2 + ((Hc-BR)**2 - hR**2)*heaviside(zR-Hc))
        FluxR = np.array([hR*uR, hR*uR**2 + PhR, hR*uR*rR])
        FluxL = np.array([hL*uL, hL*uL**2 + PhL, hL*uL*rL])
        FluxHLL = (FluxL*SR - FluxR*SL + SL*SR*(UR - UL))/(SR-SL)
        Flux = FluxHLL - (0.5*(SL+SR)/(SR-SL))*VNC
    else:
        Flux = np.zeros(3)


    return Flux, SL, SR, VNC

##################################################################
#'''----------------- Heaviside step function -----------------'''
##################################################################
def heaviside(x):
    """
    Vector-aware implemenation of the Heaviside step function.
    """
    return 1 * (x >= 0)

##################################################################
#'''--------- Compute stable timestep ---------'''
##################################################################

def time_step(U,Kk,cfl,cc2,beta,g):
### INPUT ARGS:
# U: array of variarible values at t
# Kk: grid size

### OUTPUT:
# dt: stable timestep (h>0 only)

    # signal velocties (calculated from eigenvalues)
    lam1 = abs(U[1,:]/U[0,:] - np.sqrt(cc2*beta + g*U[0,:]))
    lam2 = abs(U[1,:]/U[0,:] + np.sqrt(cc2*beta + g*U[0,:]))
    denom = np.maximum(lam1,lam2)
    
    dt = cfl*min(Kk/denom)

    return dt

##################################################################
# ZERO TOPOGRAPHY: integrate forward one time step ...
##################################################################

def step_forward_modRSW(U,U_rel,dt,Neq,Nk,Kk,Ro,alpha2,Hc,Hr,cc2,beta,g,tau_rel):
### INPUT ARGS:
# U: array of variarible values at t, size (Neq,Nk)
# dt: stable time step
# Nk, Kk: mesh info


### OUTPUT:
# UU: array of variarible values at t+1, size (Neq,Nk)

    #'''%%%----- compute extraneous forcing terms S(U) -----%%%'''
    S = np.zeros((Neq,Nk))
    S[0,:] = 0.
    S[1,:] = (1/Ro)*U[2,:]
    S[3,:] = -alpha2*U[3,:] 
    S[2,:] = -(1/Ro)*U[1,:] + (U[0,:]*U_rel[2,:]-U[2,:])/tau_rel 

    left = list(range(0,Nk))
    right = np.append(list(range(1,Nk)),0)

    hL = U[0,left] 
    hL = np.append(hL[-1],hL)
    hR = U[0,right]
    hR = np.append(hR[-1],hR)
    uL = U[1,left]/U[0,left]
    uL = np.append(uL[-1],uL)
    uR = U[1,right]/U[0,right]
    uR = np.append(uR[-1],uR)
    vL = U[2,left]/U[0,left]
    vL = np.append(vL[-1],vL)
    vR = U[2,right]/U[0,right]
    vR = np.append(vR[-1],vR)
    rL = U[3,left]/U[0,left]
    rL = np.append(rL[-1],rL)
    rR = U[3,right]/U[0,right]
    rR = np.append(rR[-1],rR)

    #''' %%%----- Determine intercell fluxes using numerical flux function -----%%%'''
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))

    for j in range(0,Nk+1):
        Flux[:,j], VNC[:,j] = NCPflux4d(hL[j],hR[j],uL[j],uR[j],vL[j],vR[j],rL[j],rR[j],Hr,Hc,cc2,beta,g)

#    for j in range(1,Nk):
#        Flux[:,j], VNC[:,j] = NCPflux4d(U[:,j-1], U[:,j],Hr,Hc,cc2,beta,g)
#    Flux[:,Nk], VNC[:,Nk] = NCPflux4d(U[:,Nk-1],U[:,0],Hr,Hc,cc2,beta,g)
#    Flux[:,0] = Flux[:,Nk]  # periodic
#    VNC[:,0] = VNC[:,Nk]  # periodic

    Pp = 0.5*VNC + Flux
    Pm = -0.5*VNC + Flux
    #'''%%%----- step forward in time -----%%%'''

    UU = U - dt*(Pp[:,1:Nk+1] - Pm[:,0:Nk])/Kk + dt*S
 
    return UU

def step_forward_modRSW_inflow(U,dt,Neq,Nk,Kk,Ro,alpha2,Hc,Hr,cc2,beta,g,U_infl):
### INPUT ARGS:
# U: array of variarible values at t, size (Neq,Nk)
# dt: stable time step
# Nk, Kk: mesh info


### OUTPUT:
# UU: array of variarible values at t+1, size (Neq,Nk)

    #'''%%%----- compute extraneous forcing terms S(U) -----%%%'''
    S = np.zeros((Neq,Nk))
    S[0,:] = 0.
    S[1,:] = (1./Ro)*U[2,:]
    S[2,:] = -(1./Ro)*U[1,:]
    S[3,:] = -alpha2*U[3,:]      

    #'''%%%----- create a copy of U which includes ghost cells -----%%%'''
    U_ext = np.zeros((Neq,Nk+2))
    U_ext[:,1:-1] = U
    U_ext[:3,0] = U_infl # specified inflow for h,hu,hv
    U_ext[:,-1] = U_ext[:,-2] # outflow condition

    #''' %%%----- Determine intercell fluxes using numerical flux function -----%%%'''
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))
    for j in range(0,Nk+1):
        Flux[:,j], VNC[:,j] = NCPflux4d(U_ext[:,j], U_ext[:,j+1],Hr,Hc,cc2,beta,g)
 
    print(VNC[3,:])

    Pp = 0.5*VNC + Flux
    Pm = -0.5*VNC + Flux
    #'''%%%----- step forward in time -----%%%'''

    UU = U - dt*(Pp[:,1:] - Pm[:,:-1])/Kk + dt*S
 
    return UU

##################################################################
# NON_ZERO TOPOGRAPHY: integrate forward one time step
##################################################################

def step_forward_topog(U,B,dt,tn,Neq,Nk,Kk,Hc,Hr,cc2,beta,alpha2,g):
    ### INPUT ARGS:
    # U: array of variarible values at t, size (Neq,Nk)
    # B: bottom topography
    # dt: stable time step
    # Nk, Kk: mesh info

    left = list(range(0,Nk))
    right = np.append(list(range(1,Nk)),0)
    
    h = U[0,:]
    h[h<1e-9] =0
    hu = U[1,:]
    hu[h<1e-9] = 0
    U[1,:] = hu
    
    Bstar = np.maximum(B[left],B[right])
    hminus = np.maximum(U[0,left] + B[left] - Bstar,0)
    hplus = np.maximum(U[0,right] + B[right] - Bstar,0)
    
    uminus = U[1,left]/U[0,left]
    uminus[np.isnan(uminus)] = 0
    uplus = U[1,right]/U[0,right]
    uplus[np.isnan(uplus)] = 0
    
    U[2,:] = np.maximum(U[2,:],0)
    rminus = U[2,left]/U[0,left]
    rminus[np.isnan(rminus)] = 0
    rplus = U[2,right]/U[0,right]
    rplus[np.isnan(rplus)] = 0
    
    huminus = hminus*uminus
    huminus = np.append(huminus[-1], huminus)
    huplus = hplus*uplus
    huplus = np.append(huplus[-1], huplus)
    
    hrminus = hminus*rminus
    hrminus = np.append(hrminus[-1], hrminus)
    hrplus = hplus*rplus
    hrplus = np.append(hrplus[-1], hrplus)
   
    hminus = np.append(hminus[-1], hminus)
    hplus = np.append(hplus[-1], hplus)
    
    Bminus = np.append(B[-1],B)
    Bplus = np.append(B[0],B[right])
    
    # reconstructed states
    Uminus = np.array([hminus, huminus, hrminus])
    Uplus = np.array([hplus, huplus, hrplus])

    
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))
    SL = np.empty(Nk+1)
    SR = np.empty(Nk+1)
    S = np.zeros((Neq,Nk))
    Sb = np.zeros((Neq,Nk))
    UU = np.empty(np.shape(U))
    
    # determine fluxes ...
    for j in range(0,Nk+1):
        Flux[:,j], SL[j], SR[j], VNC[:,j] = NCPflux_topog(Uminus[:,j],Uplus[:,j],Bminus[j],Bplus[j],Hr,Hc,cc2,beta,g)
    
    
    # compute topographic terms as per Audusse et al...
    for jj in range(0,Nk):
        
        zminus = Uminus[0,jj+1] + Bminus[jj+1]
        zplus = Uplus[0,jj] + Bplus[jj]
        
        if zminus <= Hc and zplus <= Hc:
            Sb[1,jj] = 0.5*g*(Uminus[0,jj+1]**2 - Uplus[0,jj]**2)
        elif zminus <= Hc and zplus > Hc:
            Sb[1,jj] = 0.5*g*(Uminus[0,jj+1]**2 - (Hc - Bplus[jj])**2)
        elif zminus > Hc and zplus <= Hc:
            Sb[1,jj] = 0.5*g*((Hc - Bminus[jj+1])**2 - Uplus[0,jj]**2)
        elif zminus > Hc and zplus > Hc:
            Sb[1,jj] = 0.5*g*((Hc - Bminus[jj+1])**2 - (Hc - Bplus[jj])**2)
    
#    Sb[1,:] = 0.5*g*(Uminus[0,1:]**2 - Uplus[0,:-1]**2)

    # compute extraneous forcing terms
    S[0,:] = 0
    S[1,:] = 0*U[0,:]*0.5*(np.sin(10*tn)+1)
    S[2,:] = -alpha2*U[2,:]
    
    # DG flux terms
    Pp = 0.5*VNC + Flux;
    Pm = -0.5*VNC + Flux;
    
#    # non-neg time-step:
#    dt, dt_el = dt_nonneg_SW(Uminus, Uplus, h, uminus, uplus, SL, SR, Kk)
#    dt = cfl_fc*dt

    #integrate forward to next time level
    BC = 1
    if BC == 1: #PERIODIC
        
        UU = U - dt*(Pp[:,1:] - Pm[:,:-1])/Kk + dt*Sb/Kk + dt*S

    return UU


##################################################################

def dt_nonneg_SW(Uminus, Uplus, h, uminus, uplus, SL, SR, Kk):
#%
#% Function calclates stable time step to preserve non-negativity of h as
#% per the theory ('Non-negativity preserving numerics for the SWEs').
#%
#% INPUTS:
#% > Ulstar, left reconstructed compuational state vector (hlstar, hulstar)
#% > Urstar, right reconstructed compuational state vector (hrstar, hurstar)
#% > h, standard computational state
#% > SL and SR, vector of left and right numerical speeds
#%
#% OUTPUT:
#% > dt, time step as the minimum of elemental time step
#%

    hminus = Uminus[0,:]
    hplus = Uplus[0,:]
    
    uminus = Uminus[1,:]/Uminus[0,:]
    uplus = Uplus[1,:]/Uplus[0,:]
    
    SLu = uminus - SL
    SRu = uplus - SR
    dS = SR - SL

    dt_el = np.zeros(len(h))

    for k in range(0,len(h)):

        if SL[k+1] > 0:

            denom = uminus[k+1]*hminus[k+1]
            dt_el[k] = Kk*h[k]/denom

        elif SR[k+1] < 0:

            denom = -uminus[k+1]*hplus[k+1]
            dt_el[k] = Kk*h[k]/denom

        elif SL[k+1] < 0 and SR[k+1] > 0:

            denom = SR[k+1]*SLu[k+1]*hminus[k+1]/dS[k+1]
            dt_el[k] = Kk*h[k]/denom

        elif SL[k] < 0 and SR[k] > 0:
                  
            denom = SL[k]*SRu[k]*hplus[k]/dS[k]
            dt_el[k] = Kk*h[k]/denom

        elif SL[k+1] == 0 or SR[k+1] == 0:

            dt_el[k] = 0

    dt_el[np.isnan(dt_el)]= 99999

    dt = np.min(dt_el[dt_el>1e-6])
    
    return dt, dt_el

##################################################################
# PARALLEL COMPUTING using multiprocessing
##################################################################

'''
EXAMPLE FROM:
http://blog.dominodatalab.com/simple-parallelization/

Normally you would loop over your items, processing each one:

for i in inputs
    results[i] = processInput(i)
end
// now do something with results

Alternative:

from joblib import Parallel, delayed  
import multiprocessing

# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = range(10)  
def processInput(i):  
    return i * i

num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)  

results is now [1, 4, 9 ... ]
''' 
'''
My loop:
for N in range(0,n_ens):
    tn = assim_time[index]
    
    #print 'Integrating ensemble member N =', N+1, 'of', n_ens, 'from time =', assim_time[index],' to',assim_time[index+1]

    while tn < tmeasure:
        
        dt = time_step(U[:,:,N],Kk_fc) # compute stable time step
        tn = tn + dt

        if tn > tmeasure:
            dt = dt - (tn - tmeasure) + 1e-12
            tn = tmeasure + 1e-12
        
        U[:,:,N] = step_forward_modRSW(U[:,:,N],dt,Nk_fc,Kk_fc)
        
'''
##################################################################

def ens_forecast(N, U, U_rel, q, Neq, Nk_fc, Kk_fc, cfl_fc, assim_time, index, tmeasure, dtmeasure, Hc, Hr, cc2, beta, alpha2, g, Ro, tau_rel):
    
    tn = assim_time[index]
    
    while tn < tmeasure:
        
        dt = time_step(U[:,:,N],Kk_fc,cfl_fc,cc2,beta,g) # compute stable time step
        tn = tn + dt

        if tn > tmeasure:
            dt = dt - (tn - tmeasure) + 1e-12
            tn = tmeasure + 1e-12
        
        # Inject a portion of the additive noise each timestep using an
        # Incremental Analysis Update approach (Bloom et al., 1996).
        U[:,:,N] += (q[:,N].reshape(Neq, Nk_fc)) * dt / dtmeasure
        # if hr < 0, set to zero:
        hr = U[3, :, :]
        hr[hr < 0.] = 0.
        U[3, :, :] = hr
   
        # if h < 0, set to epsilon:
        h = U[0, :, :]
        h[h < 0.] = 1e-3
        U[0, :, :] = h

        U[:,:,N] = step_forward_modRSW(U[:,:,N],U_rel,dt,Neq,Nk_fc,Kk_fc,Ro,alpha2,Hc,Hr,cc2,beta,g,tau_rel)
   
    return U[:,:,N]

##################################################################

def ens_forecast_topog(N, U, B, q, Neq, Nk_fc, Kk_fc, cfl_fc, assim_time, index, tmeasure, dtmeasure, Hc, Hr, cc2, beta, alpha2, g):

    tn = assim_time[index]

    while tn < tmeasure:
        
        dt = time_step(U[:,:,N],Kk_fc,cfl_fc,cc2,beta,g) # compute stable time step
        tn = tn + dt
        
        if tn > tmeasure:
            dt = dt - (tn - tmeasure) + 1e-12
            tn = tmeasure + 1e-12

        # Inject a portion of the additive noise each timestep using an
        # Incremental Analysis Update approach (Bloom et al., 1996).
        U[:,:,N] += (q[:,N].reshape(Neq, Nk_fc)) * dt / dtmeasure
        # if hr < 0, set to zero:
        hr = U[2, :, :]
        hr[hr < 0.] = 0.
        U[2, :, :] = hr
   
        # if h < 0, set to epsilon:
        h = U[0, :, :]
        h[h < 0.] = 1e-3
        U[0, :, :] = h
 
        U[:,:,N] = step_forward_topog(U[:,:,N],B,dt,tn,Neq,Nk_fc,Kk_fc,Hc,Hr,cc2,beta,alpha2,g)
    
    return U[:,:,N]

##################################################################




