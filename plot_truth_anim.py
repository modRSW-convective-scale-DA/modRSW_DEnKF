import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import importlib.util
import sys

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

Nk = config.Nk
outdir = config.outdir
Hc = config.Hc
Hr = config.Hr

# Set up formatting for the movie files
Writer = animation.writers['imagemagick']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

def set_frame(i):

    return U_tr[0,:,i]

U_tr = np.load(outdir+str('/test_model/U_array_Nk'+str(Nk)+'.npy'))

fig, axes = plt.subplots(4, 1, figsize=(8,8))
#axes[0].set_ylim(0.,0.4)
axes[0].hlines(Hc,-10.,Nk+10.,colors='gray',linestyles='dashed')
axes[0].hlines(Hr,-10.,Nk+10.,colors='gray',linestyles='dashed')
ims = []

for i in range(np.size(U_tr,2)):
    im0, = axes[0].plot(U_tr[0,:,i],'b')
    im1, = axes[1].plot(U_tr[1,:,i]/U_tr[0,:,i],'b')
    im2, = axes[2].plot(U_tr[2,:,i]/U_tr[0,:,i],'b')
    im3, = axes[3].plot(U_tr[3,:,i]/U_tr[0,:,i],'b')
    ims.append([im0,im1,im2,im3])

anim = animation.ArtistAnimation(fig,ims,interval=500,repeat=False)
anim.save(outdir+'/timeseries.gif',writer=writer)

plt.show()

