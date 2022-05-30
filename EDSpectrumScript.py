#%%
import EDTwoElectron2DHarmonicAtomModuleCOMandRelBasis as ed
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


N=10
numLevelstoPlot = 60
lambdaMin = 0.0001
lambdaMax = 5
numLambdaVals = 10
omgh = 1 # doesnt matter, since it all gets normalized
mStar = 1
L = np.sqrt(hbar**2/(omgh*(mStar*electronMass)))
coulombEnergy = eSquaredOvere0/(L)
lambdaVals = np.linspace(lambdaMin, lambdaMax, numLambdaVals)
epsVals = (coulombEnergy/omgh)/lambdaVals
megaEigVals_s = np.zeros((numLevelstoPlot, numLambdaVals))
megaEigVals_a = np.zeros((numLevelstoPlot, numLambdaVals))
index = 0
for eps in epsVals:
    print('eps:', eps)
    basis = ed.nonint_basis_symmetric(N, omgh)
    basis_a = ed.nonint_basis_antisymmetric(N, omgh)
    E,nRp,nRm,nrp,nrm = basis
    E_a,nRp_a,nRm_a,nrp_a,nrm_a = basis_a
    Es_s,evecs_s,Es_a,evecs_a = ed.ED_hh(basis,basis_a,omgh, mStar, eps)
    megaEigVals_s[:, index] = Es_s[0:numLevelstoPlot]
    megaEigVals_a[:, index] = Es_a[0:numLevelstoPlot]
    if index == 0:
        states = np.concatenate(basis[1:5]).flatten()
        states_a = np.concatenate(basis_a[1:5]).flatten()
        basisStatesMatrix_s = np.reshape(states, (4,np.shape(basis[0])[0]))
        basisStatesMatrix_a = np.reshape(states_a, (4,np.shape(basis_a[0])[0]))
        lArray_s = np.zeros(np.shape(evecs_s)[0])
        levelArray_s = np.zeros(np.shape(evecs_s)[0])
        lArray_a = np.zeros(np.shape(evecs_s)[0])
        levelArray_a = np.zeros(np.shape(evecs_s)[0])
        for eigenStateIndex in range(np.shape(evecs_s)[0]):
            eigenState_s = evecs_s[:, eigenStateIndex]
            basisStateIndex_s = np.argmax(eigenState_s)
            basisState_s = basisStatesMatrix_s[:, basisStateIndex_s]
            l_s = basisState_s[0] + basisState_s[2] - basisState_s[1] - basisState_s[3]
            lArray_s[eigenStateIndex] = int(l_s)
            levelArray_s[eigenStateIndex] = int(Es_s[eigenStateIndex]/omgh) - 2
        for eigenStateIndex in range(np.shape(evecs_a)[0]):
            eigenState_a = evecs_a[:, eigenStateIndex]
            basisStateIndex_a = np.argmax(eigenState_a)
            basisState_a = basisStatesMatrix_a[:, basisStateIndex_a]
            l_a = basisState_a[0] + basisState_a[2] - basisState_a[1] - basisState_a[3]
            lArray_a[eigenStateIndex] = int(l_a*(1.01))
            levelArray_a[eigenStateIndex] = int(Es_a[eigenStateIndex]/omgh) - 2
    index += 1
#HEISENBERG uncertainty principle MINIMIZATION
x = symbols('x')
cVals = np.linspace(0,lambdaMax,numLambdaVals)
solutions = np.zeros((2, numLambdaVals))
for cValIndex in range(np.shape(cVals)[0]):
    c = cVals[cValIndex]
    solutions[:, cValIndex] = np.array(list(solveset(Eq(x**4-c*x-1, 0), x, domain=Reals)))
heisenbergEnergies = (1/2)*(solutions**(-2)+solutions**(2)+2*cVals*solutions**(-1))+1


pm.produceEDSpectrumPlot(lambdaVals, megaEigVals_s, megaEigVals_a, numLevelstoPlot, heisenbergEnergies, omgh)

# %%
