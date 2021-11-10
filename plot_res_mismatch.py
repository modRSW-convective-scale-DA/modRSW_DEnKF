import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
import importlib.util
import sys

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE NUMBER 1
##################################################################
spec1 = importlib.util.spec_from_file_location("config", sys.argv[1])
config1 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(config1)

outdir1 = config1.outdir

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE NUMBER 2
##################################################################
spec2 = importlib.util.spec_from_file_location("config", sys.argv[2])
config2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(config2)

outdir2 = config2.outdir

U_400 = np.load(outdir1+'/test_model/U_array_Nk400.npy')
B_400 = np.load(outdir1+'/test_model/B_Nk400.npy')
U_200 = np.load(outdir2+'/test_model/U_array_Nk200.npy') 
B_200 = np.load(outdir2+'/test_model/B_Nk200.npy')

x_400 = np.linspace(0,1,400)
x_200 = np.linspace(0,1,200)

plt.figure(figsize=(8,4))
plt.plot(x_400,B_400,color='black')
plt.plot(x_400,U_400[0,:,1]+B_400,color='blue',label='$N_{el}^{nat}=400$')
plt.plot(x_200,U_200[0,:,1]+B_200,color='orange',label='$N_{el}=200$')
plt.xlabel('x',fontsize=20)
plt.xticks([0.0,0.25,0.5,0.75,1.0],fontsize=14)
plt.ylabel('h(x)+b(x)',fontsize=20)
plt.ylim(0,2)
plt.yticks([0.0,0.5,1.0,1.5,2.0],fontsize=14)
plt.legend(fontsize=14)
plt.show()
