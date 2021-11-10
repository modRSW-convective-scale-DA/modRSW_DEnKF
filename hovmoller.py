import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
import importlib.util
import sys

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir

### IMPORT DATA ###
U = np.load(outdir+'/U_hovplt.npy') 
x = np.arange(0.,1.,0.0025)
t = np.load(outdir+'/t_hovplt.npy')
B = np.load(outdir+'/test_model/B_Nk400.npy')

print(t.shape,x.shape)
print(t[4524])

gs = gridspec.GridSpec(2,6,height_ratios=[5,1],width_ratios=[18.5,1,18.5,1,18.5,1],wspace=0.5,hspace=0.)

# Start figure
#fig,ax = plt.subplots(2, 3, sharex=True, gridspec_kw={'height_ratios': [4,1], 'hspace':0.00},figsize=(10,5)) 

fig = plt.figure(figsize=(12,5.5))

# Top plot for density
ax1 = fig.add_subplot(gs[0])
#siglevs = np.array([0.,0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.3, 0.4, 0.5])
hlevs = np.array([0.,1.02, 1.03, 1.04, 1.05, 1.1, 1.2, 1.5, 2., 2.5])
#clcol = plt.cm.Greys(np.linspace(0.,0.5,20))
#racol = plt.cm.YlOrBr(np.linspace(0.,1.,68))
white = np.tile([1.,1.,1.,1.],(25,1))
#clcol1 = plt.cm.Reds(np.linspace(0.1,0.2,1))
#clcol2 = plt.cm.Greens(np.linspace(0.1,0.2,1))
#clcol3 = plt.cm.Oranges(np.linspace(0.1,0.2,1))
#clcol4 = plt.cm.Blues(np.linspace(0.1,0.2,1)) 
clcol = plt.cm.Greys(np.linspace(0,0.4,4))
racol = plt.cm.YlOrBr(np.linspace(0.,1.,145))
white = np.tile([1.,1.,1.,1.],(101,1))
#print(clcol)
cols = np.vstack((white,clcol,racol))
MyCmap = mpl.colors.LinearSegmentedColormap.from_list('pippo', cols)
#mapColS = list(plt.cm.Greys(np.linspace(0,0.5,100000)))
#MyCmap=colors.ListedColormap(mapColS) # make color map
cv1 = ax1.contourf(x,t[:4524:2],U[0,:,:4524:2].T,hlevs,cmap=MyCmap)
cs1 = ax1.contour(x,t[:4524:2],U[0,:,:4524:2].T,hlevs,colors='k',linewidths=0.1)
ax1bis = fig.add_subplot(gs[1])
ax1bis.set_position(pos=[0.3,0.24,0.03,0.64])
cbar = plt.colorbar(cv1,cax=ax1bis,ticks=hlevs,orientation='vertical')
cbar.ax.set_title('h',fontsize=20)
cbar.ax.tick_params(labelsize=14)
ax1.set_ylabel('t',fontsize=20,rotation=0)
ax1.tick_params(labelsize=14,labelbottom=False)
ax1.set_xticks([])
ax1.set_xticklabels([])
ax1.yaxis.set_label_coords(-0.3,0.5)
ax1below = fig.add_subplot(gs[6],sharex=ax1)
ax1below.plot(np.linspace(0,1,400),B,color='black')
ax1below.set_ylim([0,1])
ax1below.set_yticks([0,0.4,0.8])
ax1below.set_yticklabels([0,0.4,''])
ax1below.set_xticks([0.0,0.25,0.5,0.75,1.0])
ax1below.set_xticklabels([0.0,0.25,0.5,0.75,1.0])
ax1below.set_xlabel('x',fontsize=20)
ax1below.tick_params(labelsize=14)
ax1below.set_ylabel('b(x)',fontsize=20,rotation=0)
ax1below.yaxis.set_label_coords(-0.4,0.2)

# Middle plot for hor. velocity
ax2 = fig.add_subplot(gs[2])
ulevs = np.arange(1.,2.1,0.1)
cv2 = ax2.contourf(x,t[:4524:2],(U[1,:,:4524:2]/U[0,:,:4524:2]).T,ulevs,cmap=plt.cm.Reds)
cs2 = ax2.contour(x,t[:4524:2],(U[1,:,:4524:2]/U[0,:,:4524:2]).T,ulevs,colors='k',linewidths=0.5)
ax2bis = fig.add_subplot(gs[3])
ax2bis.set_position(pos=[0.574,0.24,0.03,0.64])
cbar = plt.colorbar(cv2,cax=ax2bis,ticks=ulevs,orientation='vertical')
cbar.ax.set_title('u',fontsize=20)
cbar.ax.tick_params(labelsize=14)
ax2.tick_params(labelsize=14,labelbottom=False)
ax2.set_xticks([])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2below = fig.add_subplot(gs[8],sharex=ax2)
ax2below.plot(np.linspace(0,1,400),B,color='black')
ax2below.set_ylim([0,1])
ax2below.set_yticks([0,0.4,0.8])
ax2below.set_yticklabels(['','','',''])
ax2below.set_xticks([0.0,0.25,0.5,0.75,1.0])
ax2below.set_xticklabels([0.0,0.25,0.5,0.75,1.0])
ax2below.set_xlabel('x',fontsize=20)
ax2below.tick_params(labelsize=14)

# Bottom plot for rain 
ax3 = fig.add_subplot(gs[4])
rlevs = np.array([0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06])
cm = plt.cm.get_cmap("YlGnBu")
cm.set_bad("white")
cv4 = ax3.contourf(x,t[:4524:2],np.ma.masked_where((U[2,:,:4524:2]/U[0,:,:4524:2])<5e-3,(U[2,:,:4524:2]/U[0,:,:4524:2])).T,rlevs,cmap=cm)
cs4 = ax3.contour(x,t[:4524:2],(U[2,:,:4524:2]/U[0,:,:4524:2]).T,rlevs,colors='k',linewidths=0.1)
ax3bis = fig.add_subplot(gs[5])
ax3bis.set_position(pos=[0.848,0.24,0.03,0.64])
cbar = plt.colorbar(cv4,cax=ax3bis,ticks=rlevs,orientation='vertical')
cbar.ax.set_title('r',fontsize=20)
cbar.ax.tick_params(labelsize=14)
ax3.tick_params(labelsize=14,labelbottom=False)
ax3.set_yticklabels([])
ax3.set_xticks([])
ax3.set_xticklabels([])
ax3below = fig.add_subplot(gs[10],sharex=ax3)
ax3below.plot(np.linspace(0,1,400),B,color='black')
ax3below.set_ylim([0,1])
ax3below.set_yticks([0,0.4,0.8])
ax3below.set_yticklabels(['','','',''])
ax3below.set_xticks([0.0,0.25,0.5,0.75,1.0])
ax3below.set_xticklabels([0.0,0.25,0.5,0.75,1.0])
ax3below.set_xlabel('x',fontsize=20)
ax3below.tick_params(labelsize=14)

plt.savefig(outdir+'/hovplt_truth.png')
plt.show()
