import numpy as np
import matplotlib.pyplot as plt

X = np.load('test_model/U_array_Nk800.npy')

Fr = 1.1
Ro = 0.1

h = X[0,:,:]
u = X[1,:,:]/X[0,:,:]
v = X[3,:,:]/X[0,:,:]

KE = np.empty(np.size(X,2))
KE_x = np.empty(np.size(X,2))
KE_y = np.empty(np.size(X,2))
PE = np.empty(np.size(X,2))
cor_x = np.empty(np.size(X,2))
cor_y = np.empty(np.size(X,2))
h_bar = np.empty(np.size(X,2))
u_bar = np.empty(np.size(X,2))
v_bar = np.empty(np.size(X,2))

for i in range(np.size(X,2)):
    h_bar[i] = np.average(h[:,i])
    u_bar[i] = np.average(u[:,i])
    v_bar[i] = np.average(v[:,i])
    cor_x[i] = 1./Ro*np.average(h[:,i]*v[:,i])
    cor_y[i] = 1./Ro*np.average(h[:,i]*u[:,i])
    KE[i] = 0.5*np.sum(h[:,i]*u[:,i]**2)+0.5*np.sum(h[:,i]*v[:,i]**2)
    KE_x[i] = 0.5*np.sum(h[:,i]*u[:,i]**2)
    KE_y[i] = 0.5*np.sum(h[:,i]*v[:,i]**2)
    PE[i] = 0.5*(1./Fr**2)*np.sum(h[:,i]**2)

print((KE-PE))

#h_line = plt.plot(range(np.size(X,2)),h_bar,color='green')
#u_line = plt.plot(range(np.size(X,2)),u_bar,color='purple')
#v_line = plt.plot(range(np.size(X,2)),v_bar,color='blue')
#cor_x_line, = plt.plot(range(np.size(X,2)),cor_x,color='blue')
#cor_y_line, = plt.plot(range(np.size(X,2)),cor_y,color='red')
KE_line, = plt.plot(list(range(np.size(X,2))),KE,color='blue')
KE_x_line, = plt.plot(list(range(np.size(X,2))),KE_x,color='green')
KE_y_line, = plt.plot(list(range(np.size(X,2))),KE_y,color='black')
PE_line, = plt.plot(list(range(np.size(X,2))),PE,color='orange')
TE_line, = plt.plot(list(range(np.size(X,2))),PE+KE,color='red')
plt.legend([KE_line,PE_line,TE_line],['Kinetic energy','Potential energy','Total energy'])
plt.show()
