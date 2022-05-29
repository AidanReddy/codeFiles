import numpy as np
import math
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import proplot as pplt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
mpl.rcParams['text.usetex']= True
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams["savefig.jpeg_quality"]
pplt.rc['grid'] = False

figSaveDir='/Users/aidanreddy/Desktop/TwistedTMD/plots/'


plt.savefig('moireAtomGTLineCut%stheta=%dNu=%dEpsilonED=%dd=%d=%dTMin=%dTMax=%dnumTVals=%dN=%dIncluded=%d' % (date, theta, nu, epsForED, dVals[0], 0, TLineCutMin, TLineCutMax, numTLineCutVals, N, numEStatesIncluded))
