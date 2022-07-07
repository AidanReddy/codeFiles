#%%
import EDandMatrixElementsModule as ed
import continuumModelBandsModule as cmb
import numpy as np
import scipy as sp
from sympy import *
import mpmath as mp
import math
from numba import njit
import plotModule as pm

#physical constants
hbar = 6.582 * 10**(-13) # meV * s
#static, bulk dielectric tensor components of hBN from npj 2D Materials and Applications (2018) 2:6 ; doi:10.1038/s41699-018-0050-x
epsilonPerp = 3.76
epsilonPar = 6.93
epsilonEff = math.sqrt(epsilonPar*epsilonPerp)
epsilonEffD = math.sqrt(epsilonPar/epsilonPerp)
dielectricConstant = epsilonEff
electronMass = 5.856301 * 10**(-29) # meV *(second/Å)
eSquaredOvere0 =  14400 #meV * angstrom
WattSecondsPermeV = 1.602 * 10**(-22)
Kb = 0.08617 # meV per Kelvin
SquareMetersPerSqareAngstrom = 10**(-20)
JoulesPermeV = 1.602 * 10**(-22)


N=12
numLevelstoPlot = 60
lambdaMin = 0.0001
lambdaMax = 5
numLambdaVals = 100
omgh = 1 # doesnt matter, since it all gets normalized
mStar = 1
L = np.sqrt(hbar**2/(omgh*(mStar*electronMass)))
coulombEnergy = eSquaredOvere0/(L)
lambdaVals = np.linspace(lambdaMin, lambdaMax, numLambdaVals)
epsVals = (coulombEnergy/omgh)/lambdaVals
megaEigVals_s = np.zeros((numLevelstoPlot, numLambdaVals))
megaEigVals_a = np.zeros((numLevelstoPlot, numLambdaVals))
E0, stateList, E0_s, stateList_s, E0_a, stateList_a = ed.nonint_basis(N, omgh)
index = 0

#print('stateList:', stateList_a[0:7])
print(stateList_s[0:7,2]+stateList_s[0:7,3])
for eps in epsVals:
    #print('eps:', eps)
    Es, Es_s,evecs_s,Es_a,evecs_a, parityList = ed.ED_hh(E0_s, E0_a, stateList_s, stateList_a, omgh, mStar,eps)
    megaEigVals_s[:, index] = Es_s[0:numLevelstoPlot]
    megaEigVals_a[:, index] = Es_a[0:numLevelstoPlot]
    index += 1
#HEISENBERG uncertainty principle MINIMIZATION
x = symbols('x')
cVals = np.sqrt(np.pi)*np.linspace(0,lambdaMax,numLambdaVals)
solutions = np.zeros((2, numLambdaVals))
for cValIndex in range(np.shape(cVals)[0]):
    c = cVals[cValIndex]
    solutions[:, cValIndex] = np.array(list(solveset(Eq(x**4-c*x-1, 0), x, domain=Reals)))
heisenbergEnergies = (1/2)*(solutions**(-2)+solutions**(2)+2*cVals*solutions**(-1))+1

pm.produceEDSpectrumPlot(lambdaVals, megaEigVals_s, megaEigVals_a, numLevelstoPlot, heisenbergEnergies, omgh)

# %%
