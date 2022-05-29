import numpy as np
import scipy.linalg as la
import numpy.linalg #I also import this because numba support numpy linalg but not necessarily scipy linalg functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import proplot as pplt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams["savefig.jpeg_quality"]
import math
import timeit
import multiprocessing as mp
from numba import njit
from datetime import date
today = date.today()
date = today.strftime("%b%d%Y")

import continuumModelBandsModule as cmb

N = 7 #7 ### Creates NxN k space mesh. N MUST BE ODD!!!!!
numShells = 4 # number of reciprocal lattice vector shells
theta = 2 * math.pi/180 # twist angle
nu = 2
modStrengthFactor = 1
material = 'MoS2' #WS2
hbareta =  1 # meV
hbarwMin = 0 # meV
hbarwMax = theta * 180/math.pi * modStrengthFactor * 200  #meV
numhbarwVals = 500
TVals = np.linspace(10,1000,100)
dMin = 0
dMax = 20
numdVals = 2
dVals = np.linspace(dMin, dMax, numdVals) #np.linspace(1,1)
numdVals = np.shape(dVals)[0]
dLineCutMin = np.log10(0.01)
dLineCutMax = np.log10(40)
numdLineCutVals = 20
dLineCutVals = np.logspace(dLineCutMin,dLineCutMax,numdLineCutVals)
TMin = np.min(TVals)
TMax = np.max(TVals)
numTVals = np.shape(TVals)[0]
hbarwVals = np.linspace(hbarwMin, hbarwMax, numhbarwVals)
electronMass = 5.856301 * 10**(-29) # meV *(second/Å)

if material == 'WS2':
    a = 3.18 # atomic lattice constant in Å
    V1 = 33.5 * modStrengthFactor #meV
    V2 = 4.0 * modStrengthFactor #meV
    V3 = 5.5 * modStrengthFactor #meV
    phi = math.pi #phase that enters moiré potential Fourier expansion
    mStar = 0.87 * electronMass # effective mass at untwisted bilayer valence band maximum
if material =='MoS2':
    a = 3.182 # atomic lattice constant in Å
    V1 = 39.45 * modStrengthFactor #meV
    V2 = 6.5 * modStrengthFactor #meV
    V3 = 10.0 * modStrengthFactor #meV
    phi = np.pi #phase that enters moiré potential Fourier expansion
    electronMass = 5.856301 * 10**(-29) # meV *(second/Å)
    mStar = 0.9 * electronMass # effective mass at untwisted bilayer valence band maximum

hbar = 6.582 * 10**(-13) # meV * s
#static, bulk dielectric tensor components of hBN from npj 2D Materials and Applications (2018) 2:6 ; doi:10.1038/s41699-018-0050-x
epsilonPerp = 3.76
epsilonPar = 6.93
epsilonEff = math.sqrt(epsilonPar*epsilonPerp)
epsilonEffD = math.sqrt(epsilonPar/epsilonPerp)
dielectricConstant = epsilonEff

eSquaredOvere =  14400 * 1/dielectricConstant #meV * angstrom #CGS
WattSecondsPermeV = 1.602 * 10**(-22)
Kb = 0.08617 # meV per Kelvin
SquareMetersPerSqareAngstrom = 10**(-20)

am = a/theta #moiré period in linear approximation
A = (am**2) * math.sqrt(3)/2 * N**2 # define total lattice area A
b1 = (4*math.pi/math.sqrt(3)) * (1/am) * np.array([1,0]) # define reciprocal basis vectors b1 and b2 for moire lattice
b2 = (4*math.pi/math.sqrt(3)) * (1/am) * np.array([0.5, math.sqrt(3)/2])
a1 = am * np.array([math.sqrt(3)/2, -1/2]) # define real basis vectors a1 and a2 for moire lattice
a2 = am * np.array([0, 1])
gamma = np.array([0,0]) ### symmetry points in 1BZ
K = ((b1+b2)/la.norm(b1+b2)) * (la.norm(b1)/math.sqrt(3))
KPrime = ((2*b2 - b1)/la.norm(2*b2 - b1)) * (la.norm(b1)/math.sqrt(3))
KPrime2 = -1 * K


#USE THIS REGION IF PROCESSING AN ALREADY CALCULATED MEGAMATRIX
computeMegaMatrixFromScratch = True

#megaMatrix = np.load('megaMatrixLocalMoS2Apr202022N=7Shells=4Hbareta=1.00Theta=2.0nu=2energy0,400,500Ef=-2TVals=10,990,50ModFactor=1Epsilon=5.104586173236768.npy')
#megaMatrix = np.load('megaMatrixLocalMoS2Apr132022N=7Shells=8Hbareta=1.00Theta=1.0energy0,200,500Ef=-12TVals=10,990,50ModFactor=1Epsilon=5.104586173236768.npy')
#megaMatrix=np.load('megaMatrixLocalMoS2Apr092022N=7Shells=6Hbareta=1.00Theta=1.0energy0,200,500Ef=-12TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')f
#megaMatrix=np.load('megaMatrixLocalMoS2Mar182022N=7Shells=4Hbareta=1.00Theta=3.0energy0,1200,500Ef=-56TVals=10,1000,100ModFactor=2Epsilon=5.104586173236768.npy')
#megaMatrix = np.load('megaMatrixLocalMoS2Mar312022N=7Shells=4Hbareta=1.00Theta=2.0energy0,400,500Ef=-26TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')
#megaMatrix = np.load('megaMatrixLocalMoS2Mar312022N=7Shells=4Hbareta=1.00Theta=1.0energy0,300,500Ef=-14TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')
#megaMatrix=np.load('megaMatrixLocalMoS2Mar052022N=7Shells=4Hbareta=1.00Theta=4.0energy0,800,500Ef=-81TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')
#megaMatrix = np.load('megaMatrixWS2JupyterMarch3N=7Shells=4Hbareta=1.00Theta=3.0energy0,700,400Ef=-50TVals=10,1000,199ModFactor=1Epsilon=5.104586173236768.npy')
#megaMatrix = np.load('megaMatrixJupyterJanuary10N=7Shells=4Hbareta=0.20Theta=2.0energy0,300,301Ef=-25TVals=10,1000,199ModFactor=1Epsilon=5.104586173236768.npy')
#megaMatrix = np.load('megaMatrixJupyterJanuary11N=7Shells=4Hbareta=0.10Theta=1.0energy0,300,301Ef=-12TVals=10,1000,199ModFactor=1Epsilon=5.104586173236768.npy', mmap_mode='r')
#megaMatrix = np.load('megaMatrixJupyterJanuary10N=7Shells=4Hbareta=0.20Theta=2.0energy0,300,301Ef=-25TVals=10,1000,199ModFactor=1Epsilon=5.104586173236768.npy', mmap_mode='r')

#print('megaMatrixShape:', np.shape(megaMatrix))

"""
megaMatrixSparse = megaMatrix[:,:,::2,:,:] #takes in a megamatrix that has more T values for unscreened calculation and de-densifies it to be able to do a screened calculation

del megaMatrix

megaMatrix = megaMatrixSparse
"""

TValsSparse = TVals[::2]

del TVals

TVals = TValsSparse

TMin = np.min(TVals)

TMax = np.max(TVals)

numTVals = np.shape(TVals)[0]

# define gVals
shells = cmb.computeShell(numShells, am)
numInShells = np.shape(shells)[0]
gVals = shells#computeShell(math.floor(numShells/2))
numgVals = np.shape(gVals)[0]
numBands = np.shape(shells)[0]
firstAndSecondShells = cmb.computeShell(2, am) # for use in findgVals

#mesh
mesh, reducedMesh, reducedMeshCounter, meshToReducedMeshIndexMap = cmb.computeMesh(N,am)
numMesh = np.shape(mesh)[0]
numRedMesh = np.shape(reducedMesh)[0]

@njit
def sendToMesh(q, mesh=mesh, numMesh=numMesh, K=K, firstAndSecondShells=firstAndSecondShells, numTries = 0):
    if numTries == 0:
        offset = np.array([0,0]).astype(np.float64)
    if numTries != 0:
        offset = firstAndSecondShells[numTries-1]
    qOffset = q.astype(np.float64)-offset
    for i in range(numMesh):
        distance = np.linalg.norm(mesh[i]-qOffset)
        if distance <= (1/100)*(np.linalg.norm(K)/numMesh):
            numTries = 0
            return(mesh[i], i, offset)
    else:
        numTries += 1
        return(sendToMesh(q, mesh, numMesh, K, firstAndSecondShells, numTries))

#kronecker delta
def kronDelta(a,b):
    if np.array_equal(a,b):
        return(1)
    else:
        return(0)

#modulation potential
def computeV(b, V1 = V1, V2 = V2, V3 = V3, phi = phi, gVectors = shells, b1 = b1):
    if b in gVectors:
        if np.allclose(la.norm(b), la.norm(b1)):
            V = V1 * np.exp(complex(0,phi))
            return(V)
        elif np.allclose(la.norm(b), 1.73205081*la.norm(b1)):
            V = V2 * np.exp(complex(0,phi))
            return(V)
        elif np.allclose(la.norm(b), 2*la.norm(b1)):
            V = V3 * np.exp(complex(0,phi))
            return(V)
    return(0)

#compute matrix elements
def computeMatrixElement(k, g, gprime):
    V = computeV(gprime - g)
    matrixElement = -1 * kronDelta(g, gprime) * (hbar)**2 *(1/(2*mStar)) * np.dot(k+g, k+g) + V
    return(matrixElement)

# compute matrix
def computeMatrix(k, gVals = shells):
    gprimeVals = gVals
    matrix = np.array([])
    for i in gVals:
        for j in gprimeVals:
            matrixElement = computeMatrixElement(k,i,j)
            matrix = np.append(matrix, matrixElement)
    matrix = matrix.reshape((len(gVals)),len(gprimeVals))
    return(matrix)

def computeEigStuff(k):
    eigVals, eigVecs = la.eig(computeMatrix(k))
    eigValsSorted = np.sort(eigVals)
    eigVecsSorted = eigVecs[:, eigVals.argsort()]
    return(eigValsSorted, eigVecsSorted)

#old version, it is proven to work 01/04/22
def computeMegaEigStuff(reducedMesh = reducedMesh, numBands = numBands):
    #the mesh used here is the reducedMesh
    numRedMesh = np.shape(reducedMesh)[0]
    megaEigValArray = np.array([])
    megaEigVecArray = np.array([])
    for i in range(numRedMesh):
        redMeshVal = reducedMesh[i]
        redMeshValEigVals, redMeshValEigVecs = computeEigStuff(redMeshVal)
        megaEigValArray = np.append(megaEigValArray, redMeshValEigVals)
        megaEigVecArray = np.append(megaEigVecArray, redMeshValEigVecs)
    megaEigValArray = megaEigValArray.reshape(numRedMesh, numBands)
    megaEigVecArray = megaEigVecArray.reshape(numRedMesh, numBands, numBands)
    return(np.real(megaEigValArray), megaEigVecArray)


#compute megaEigStuff only for reducedMesh
megaEigValArray, megaEigVecArray = computeMegaEigStuff()
print('eigstuff done')

#set VBM to zero
megaEigValArray = megaEigValArray - np.max(megaEigValArray)

print('megaEigValArray shape:', np.shape(megaEigValArray))
firstBand = megaEigValArray[:, numBands-1]
secondBand = megaEigValArray[:, numBands-2]
thirdBand = megaEigValArray[:, numBands-3]
fourthBand = megaEigValArray[:, numBands-4]
fifthBand = megaEigValArray[:, numBands-5]
sixthBand = megaEigValArray[:, numBands-6]

print('reduced mesh counter:', reducedMeshCounter)
print('first band:', np.shape(firstBand))
bottomFirstBand = np.min(firstBand)
bottomSecondBand = np.min(secondBand)
#note we are computing weighted average
centerFirstBand = np.average(firstBand,0,reducedMeshCounter)
centerSecondBand = np.average(secondBand,0,reducedMeshCounter)
centerThirdBand = np.average(thirdBand,0,reducedMeshCounter)
centerFourthBand = np.average(fourthBand,0,reducedMeshCounter)
centerFifthBand = np.average(fifthBand,0,reducedMeshCounter)
centerSixthBand = np.average(sixthBand,0,reducedMeshCounter)
bandgap = np.real((centerFirstBand+centerSecondBand)/2 - (centerThirdBand+centerFourthBand+centerFifthBand+centerSixthBand)/4)

#UPDATED AUGUST 21 2021
if nu == 4:
    Ef = np.real((centerFirstBand+centerSecondBand)/2 + (centerThirdBand+centerFourthBand+centerFifthBand+centerSixthBand)/4)/2
if nu == 2:
    Ef = bottomFirstBand

print('Ef:', Ef)
print('first band bottom to top width:', np.max(firstBand)-np.min(firstBand ))

#NUmba version, which I needed to modify because numba doesnt support np.allclose(), runs several times faster than the python version!
@njit
def findgIndex(g, shells = shells):
    lenShells = np.shape(shells)[0]
    gIndex = 0
    for gVal in shells:
        #USE THIS IF NOT USING @njit: if np.allclose(g, gVal):
        if np.linalg.norm(g-gVal) <= np.linalg.norm(g)/100: #FOR USE WITH NJIT
            return(gIndex)
        gIndex += 1
    return(1000)
    
@njit
def fermi(E, T = 0, mu = Ef, Kb = Kb):
    if T == 0:
        if E <= mu:
            return(1)
        if E > mu:
            return(0)
    else:
        fermi = (np.exp(np.real((E - mu)/(Kb*T))) + 1)**(-1)
        return(fermi)


@njit
def MuTtoN(mu, T, megaEigValArray = megaEigValArray, reducedMeshCounter = reducedMeshCounter):
    n = 0
    #update N calculation to account for symmetry reduction weighting
    for i in range(np.shape(megaEigValArray)[0]):
        for j in range(np.shape(megaEigValArray)[1]):
            energyVal = megaEigValArray[i,j]
            n += fermi(energyVal, T, mu) * reducedMeshCounter[i]
    return(n)

@njit
def computeMuOfTVector(Ef, TVals = TVals, megaEigValArray = megaEigValArray, reducedMeshCounter = reducedMeshCounter):
    numTVals = np.shape(TVals)[0]
    n = 0
    for i in range(np.shape(megaEigValArray)[0]):
        for j in range(np.shape(megaEigValArray)[1]):
            energyVal = megaEigValArray[i,j]
            if energyVal <= Ef:
                n += reducedMeshCounter[i]
    MuOfTVector = np.zeros(numTVals)
    trialMu = Ef
    for TIndex, T in np.ndenumerate(TVals):
        nFound = False
        while nFound == False:
            trialn = MuTtoN(trialMu, T)
            nFound = abs(n-trialn)/n < 0.01#0.001 # made lece precise 04/18/2022 to try to get nu=2 calculation to work
            trialMu += 0.01
        MuOfTVector[TIndex] = trialMu
    return(MuOfTVector)

print('starting muOfT')

MuOfTVector = computeMuOfTVector(Ef)

MuOfTVectorName = str('MuOfTVector%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dEf=%dTVals=%d,%d,%dModFactor=%sEpsilon=%s' % (material, date, N, numShells, hbareta, theta*(180/math.pi),nu, hbarwMin, hbarwMax, numhbarwVals, Ef, TMin,TMax, numTVals, modStrengthFactor, dielectricConstant))
np.save(MuOfTVectorName, MuOfTVector)


print('muOfT done')

@njit#(cache = True) #cache = True saves numba compilation for later
def computeDDRFMatrix(q, hbarwVals = hbarwVals, TVals = TVals, mesh = mesh, meshToReducedMeshIndexMap = meshToReducedMeshIndexMap, hbareta = hbareta, gVals = gVals, numBands = numBands, megaEigValArray = megaEigValArray, megaEigVecArray = megaEigVecArray):
    DDRFMatrix = np.zeros((numhbarwVals, numTVals, numBands, numBands), dtype=np.complex128) #for some reason the np.complex128 is crucial for numba compatibility
    for kIndex in range(numMesh):
        print('kIndex:', kIndex)
        k = mesh[kIndex]
        kPlusQ = k + q
        kPlusQInMesh, kPlusQInMeshIndex, KTrans = sendToMesh(kPlusQ)
        kRedIndex = meshToReducedMeshIndexMap[kIndex]
        kEigVals = megaEigValArray[kRedIndex]
        kPlusQInMeshRedIndex = meshToReducedMeshIndexMap[kPlusQInMeshIndex]
        kPlusQInMeshEigVals = megaEigValArray[kPlusQInMeshRedIndex]
        kEigVecs = megaEigVecArray[kRedIndex]
        kPlusQInMeshEigVecs = megaEigVecArray[kPlusQInMeshRedIndex]
        for n in range(numBands):
            Enk =  kEigVals[n]
            nkEigVec = kEigVecs[:, n]
            for nPrime in range(numBands):
                EnPrimekPlusQ = kPlusQInMeshEigVals[nPrime]
                nPrimekPlusQEigVec = kPlusQInMeshEigVecs[:, nPrime]
                deltaEnergy = EnPrimekPlusQ - Enk
                innerProductList = np.zeros(numgVals, dtype = np.complex128) #for some reason the np.complex128 is crucial for numba compatibility
                for KIndex in range(numgVals):
                    K = gVals[KIndex]
                    ZKkn = nkEigVec[KIndex]
                    for gIndex in range(numgVals):
                        g = gVals[gIndex]
                        KPlusGPlusKTrans =  K + g + KTrans
                        KPlusGPlusKTransIndex = findgIndex(KPlusGPlusKTrans)
                        if KPlusGPlusKTransIndex != 1000: # I have my findgINdex function set up so that if its argument is not includes in my shells, the function returns False, and then I just don't count that term in my sum.
                            ZKPlusGPlusKTranskPlusqnPrime = nPrimekPlusQEigVec[KPlusGPlusKTransIndex]
                            ZKPlusGPlusKTranskPlusqnPrimeStar = np.conj(ZKPlusGPlusKTranskPlusqnPrime)
                            #not acutally an inner product, just one component of an inner product
                            innerProduct = ZKPlusGPlusKTranskPlusqnPrimeStar * ZKkn
                            innerProductList[gIndex] += innerProduct
                innerProductMatrix = np.outer(np.conj(innerProductList), innerProductList)
                for hbarwIndex in range(numhbarwVals):
                    hbarw = hbarwVals[hbarwIndex]
                    otherTerm = complex(hbarw - deltaEnergy, -hbareta)/((hbarw - deltaEnergy)**2 + hbareta**2)
                    for TIndex, T in np.ndenumerate(TVals):
                        MuOfT = MuOfTVector[TIndex]
                        fermiTerm = fermi(Enk, T, MuOfT) - fermi(EnPrimekPlusQ, T, MuOfT)
                        incrementMatrix = innerProductMatrix * fermiTerm * otherTerm
                        DDRFMatrix[hbarwIndex][TIndex] += incrementMatrix
    DDRFMatrix *= 2/A # FACTOR OF TWO ACCOUNTS FOR SPIN DEGENERACY
    return(DDRFMatrix)

if computeMegaMatrixFromScratch == True:
    megaMatrix = np.zeros([numRedMesh, numhbarwVals, numTVals, numBands, numBands], dtype=complex)
    for qIndex in range(numRedMesh):
        start = timeit.default_timer()
        q = reducedMesh[qIndex]
        megaMatrix[qIndex] += computeDDRFMatrix(q)
        stop = timeit.default_timer()
        print('\n time for q value:', stop-start)
    megaMatrixName = str('megaMatrixLocal%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dEf=%dTVals=%d,%d,%dModFactor=%sEpsilon=%s' % (material, date, N, numShells, hbareta, theta*(180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, Ef, TMin,TMax, numTVals, modStrengthFactor, dielectricConstant))
    np.save(megaMatrixName, megaMatrix)

###Taken from jupyter file for comparison

def constructU(megaMatrix = megaMatrix, dVals = dVals, TVals = TVals, reducedMesh = reducedMesh, shells = shells, hbarwVals = hbarwVals, A = A):
    numdVals = np.shape(dVals)[0]
    gDiagonal = np.eye(numgVals)
    gqDiagonal = np.tile(gDiagonal, (numRedMesh, 1,1)) #(q,g,gPrime)
    qPlusgNorm = np.zeros((numRedMesh, numgVals))
    for qIndex in range(numRedMesh):
        for gIndex in range(numgVals):
            qPlusgNorm[qIndex, gIndex] += np.linalg.norm(reducedMesh[qIndex] + gVals[gIndex])
    gqDiagonalTimesqPlusGNorm = np.einsum('ij, ijk->ijk', qPlusgNorm, gqDiagonal)
    gqDiagonalTimesqPlusGNormdVals = np.tile(gqDiagonalTimesqPlusGNorm, (numdVals, 1,1,1)) # (d,q,g,gPrime)
    gqDiagonalTimesqPlusGNormdValsTimesdEff = -1 * epsilonEffD * np.einsum('i,ijkl->ijkl', dVals, gqDiagonalTimesqPlusGNormdVals)
    exponentialDecayFactor = np.exp(gqDiagonalTimesqPlusGNormdValsTimesdEff)
    ones = np.ones((numdVals, numRedMesh,numgVals,numgVals))
    UDiagonal = (2*np.pi)*eSquaredOvere*np.divide(ones, gqDiagonalTimesqPlusGNormdVals, out=np.zeros_like(ones), where=gqDiagonalTimesqPlusGNormdVals!=0) #when dividing the two matrcies, set elements that are divides by zero to zero
    UOffDiagonal=exponentialDecayFactor*UDiagonal
    U = np.block([[UDiagonal, UOffDiagonal], [UOffDiagonal, UDiagonal]]) #(d, q, g, gPrime)
    return(U)

def computeThermalFactorArray(TVals):
    betaVals = 1/(Kb*TVals)
    numTVals=np.shape(TVals)[0]
    hbarwBroadcast = np.transpose(np.tile(hbarwVals, (numTVals, 1)))
    betahbarw = np.einsum('ij,j->ij', hbarwBroadcast, betaVals) #(hw, beta)
    thermalExponent = np.exp(betahbarw)
    ones = np.ones((numhbarwVals, numTVals))
    thermalExponentMinusOne = thermalExponent-1
    boseFactor = np.divide(ones, thermalExponentMinusOne, out=np.zeros_like(ones), where=thermalExponentMinusOne!=0) #when dividing the two matrcies, set elements that are divides by zero to zero
    boseFactorDifferenceRedundant = np.subtract.outer(boseFactor,boseFactor) #(w, beta1, wPrime, beta2)
    boseFactorDifferenceOutOfOrder = np.diagonal(boseFactorDifferenceRedundant, offset=0, axis1=0, axis2=2) #(beta1, beta2, w)
    boseFactorDifference = np.einsum('kji->ijk', boseFactorDifferenceOutOfOrder) #(w, beta1,beta2)
    thermalFactor = np.einsum('i, ijk->ijk', hbarwVals, boseFactorDifference)
    return(thermalFactor)

thermalFactor = computeThermalFactorArray(TVals)

def computePMatrixIntegrandVectorized(megaMatrix, dVals, TVals, thermalFactor):
    print('pre U')
    U = constructU(megaMatrix, dVals, TVals)
    #thermalFactor = computeThermalFactorArray()
    print('post U')
    AHMegaMatrix = (megaMatrix - np.conj(np.einsum('...ij->...ji', megaMatrix)))/2 #(q,w,T,g,gPrime)
    print(numRedMesh)
    print(np.shape(AHMegaMatrix), np.shape(U[:,:,0:numgVals, numgVals:2*numgVals]))
    pathChiU = np.einsum_path('ijklm,nimo->nijklo', AHMegaMatrix, U[:,:,0:numgVals, numgVals:2*numgVals], optimize='optimal')[0] #determined optimal contraction path
    start = timeit.default_timer()
    AHChi1U = np.einsum('ijklm,nimo->nijklo', AHMegaMatrix, U[:,:, 0:numgVals, numgVals:2*numgVals], optimize=pathChiU) #(d,q,omega,T,g,gPrime). mutlitply in this way to avoid explicity constructing the block matrix
    stop = timeit.default_timer()
    print('Chi1U time:', stop-start)
    start = timeit.default_timer()
    AHChi2UDagger = np.einsum('ijklm,nimo->nijklo', AHMegaMatrix, U[:,:,numgVals:2*numgVals, 0:numgVals], optimize=pathChiU)
    stop = timeit.default_timer()
    print('AHChi2UDagger time:', stop-start)
    print('AHChi1UShape:', np.shape(AHChi1U))
    print('AHChi2UDaggerShape:', np.shape(AHChi2UDagger))
    pathTrace = np.einsum_path('ijklmn,ijkonm -> ijklo', AHChi1U, AHChi2UDagger, optimize='optimal')[0]
    trace = np.einsum('ijklmn,ijkonm -> ijklo', AHChi1U, AHChi2UDagger, optimize=pathTrace) #(d,q,omega,T1,T2)
    print('trace done!')
    print('shape trace:', np.shape(trace))
    PMatrixIntegrand = np.einsum('ijk, lmijk -> lmijk', thermalFactor, trace) * 2 * (math.pi*hbar)**(-1) * (A*(((N**2)-1)/N**2))**(-1) * (SquareMetersPerSqareAngstrom)**(-1) * WattSecondsPermeV
    return(PMatrixIntegrand)

def PArrayGIntegrandWRTEnergyGArray(TVals, PMatrixIntegrand):
    dhbarw = hbarwVals[1]-hbarwVals[0]
    PMatrixIntegratedOverQ = np.einsum('ijlkm, j -> ilkm', PMatrixIntegrand, reducedMeshCounter)
    PArray = np.sum(PMatrixIntegratedOverQ, axis = 1)*dhbarw #(d,T1,T2)
    DeltaT = np.subtract.outer(TVals, TVals)
    ones = np.ones_like(DeltaT)
    OneOverDeltaT = np.divide(ones, DeltaT, out=np.zeros_like(ones), where=DeltaT!=0)
    GIntegrandWRTEnergy = np.einsum('ij, klij -> klij', OneOverDeltaT, PMatrixIntegratedOverQ)
    GArray = -PArray*OneOverDeltaT
    for i in range(np.shape(GArray)[1]): #this function averages the diagonal
        if i==0:
            GArray[:,i,i] = GArray[:,i, i+1]
        if i==(np.shape(GArray)[1]-1):
            GArray[:,i,i] = GArray[:,i, i-1]
        else:
            GArray[:,i,i] = (GArray[:,i, i-1] + GArray[:,i, i+1])/2
    return(PArray, GIntegrandWRTEnergy, GArray)


PMatrixIntegrand = computePMatrixIntegrandVectorized(megaMatrix, dVals, TVals, thermalFactor)

PArray, GIntegrandWRTEnergy, GArray = PArrayGIntegrandWRTEnergyGArray(TVals, PMatrixIntegrand)
PArray=np.abs(np.real(PArray))
GIntegrandWRTEnergy = np.abs(np.real(GIntegrandWRTEnergy))
GArray = np.abs(np.real(GArray))
PArrayName = str('PArrayLocalUNSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals))
np.save(PArrayName, np.abs(np.real(PArray)))
GIntegrandWRTEnergyName = str('GIntegrandWRTEnergyLocalUNSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals))
np.save(GIntegrandWRTEnergyName, np.abs(np.real(GIntegrandWRTEnergy)))
GArrayName = str('GArrayLocalUNSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals))
np.save(GArrayName, np.abs(np.real(GArray)))

#LineCutInD

T1Index, T2Index = np.unravel_index(np.abs(GArray[0]).argmax(), GArray[0].shape)

print('T1Index, T2Index:', T1Index, T2Index)

megaMatrixVal = megaMatrix[:,:,T2Index:T2Index+2]

TValsFordLineCut = np.array([TVals[T1Index], TVals[T2Index]])

GLineCutInd = np.zeros_like(dLineCutVals)

thermalFactor = computeThermalFactorArray(TValsFordLineCut)

PIntegrand = computePMatrixIntegrandVectorized(megaMatrixVal, dLineCutVals, TValsFordLineCut, thermalFactor)

print(np.shape(PIntegrand))
Pofd, GIntegrandofd, Gofd = PArrayGIntegrandWRTEnergyGArray(TValsFordLineCut, PIntegrand)
print(np.shape(Gofd))
GLineCutInd = -np.real(Gofd[:,1,0])
print(GLineCutInd)
GLineCutIndName = str('GLineCutIndLocalLogUNSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%dModFactor=%sdVals=%d,%d,%d' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TVals[T1Index], TVals[T2Index], modStrengthFactor, dLineCutMin, dLineCutMax, numdLineCutVals))
np.save(GLineCutIndName, GLineCutInd)

"""
PLOTS
"""

# G color plot

dIndex = 0

fig, ax = plt.subplots()

imshow1 = ax.imshow(GArray[dIndex]*10**(-6), cmap = 'hot', origin = 'lower', extent = [10,1000,10,1000])

cbar1 = plt.colorbar(imshow1)

cbar1.ax.set_xlabel(r'$G_{12}$ $(\frac{MW}{m^2K})$', loc='left')

ax.set_aspect('equal', adjustable='box')
ax.set_xticks([10,250,500,750,1000])
ax.set_yticks([10,250,500,750,1000])
ax.minorticks_off()

theta1Temp1, theta1Temp2 = np.unravel_index(np.abs(GArray[dIndex]).argmax(), GArray[dIndex].shape)

ax.set_title(r'$\theta=%d^{\circ}$' % (theta*180/math.pi), y=1.0, x=0.8,pad=-14, color='white')

ax.set(xlabel = r'T$_{1}$ (K)', ylabel=r'T$_{2}$ (K)')

plt.savefig('GArrayColorPlotLocalUNSCREENEDdIndex=%d%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (dIndex, material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals), bbox_inches='tight')

dIndex = 1

fig, ax = plt.subplots()

imshow1 = ax.imshow(GArray[dIndex]*10**(-6), cmap = 'hot', origin = 'lower', extent = [10,1000,10,1000])

cbar1 = plt.colorbar(imshow1)

cbar1.ax.set_xlabel(r'$G_{12}$ $(\frac{MW}{m^2K})$', loc='left')

ax.set_aspect('equal', adjustable='box')
ax.set_xticks([10,250,500,750,1000])
ax.set_yticks([10,250,500,750,1000])
ax.minorticks_off()

theta1Temp1, theta1Temp2 = np.unravel_index(np.abs(GArray[dIndex]).argmax(), GArray[dIndex].shape)

ax.set_title(r'$\theta=%d^{\circ}$' % (theta*180/math.pi) , y=1.0, x=0.8,pad=-14, color='white')

ax.set(xlabel = r'T$_{1}$ (K)', ylabel=r'T$_{2}$ (K)')

plt.savefig('GArrayColorPlotLocalUNSCREENEDdIndex=%d%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (dIndex, material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals), bbox_inches='tight')

# G integrand plot

dIndex = 0

fig, ax = plt.subplots()

theta1Temp1, theta1Temp2 = np.unravel_index(np.abs(GArray[dIndex]).argmax(), GArray[dIndex].shape)

ax.plot(hbarwVals, 6.242*10**(21)*GIntegrandWRTEnergy[dIndex,:, theta1Temp1,theta1Temp2]*10**(-6), color = 'black', linewidth='1')

ax.set_title(r'$\theta=%d^{\circ}$, T = %d K' % ((theta* 180/math.pi), TVals[theta1Temp2]), y=1.15, x=0.63,pad=-14)

ax.set(xlabel = r'$\hbar\omega$ (meV)', ylabel=r'$f(\omega)$ (s$^{-1}$m$^{-2}$K$^{-1}$)')
ax.tick_params(which='both',direction='in')
ax.margins(0)
plt.savefig('GIntegrandWRTEnergyLocalUNSCREENEDdIndex=%d%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (dIndex, material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals), bbox_inches='tight')

dIndex = 1

fig, ax = plt.subplots()

theta1Temp1, theta1Temp2 = np.unravel_index(np.abs(GArray[dIndex]).argmax(), GArray[dIndex].shape)

ax.plot(hbarwVals, 6.242*10**(21)*GIntegrandWRTEnergy[dIndex,:, theta1Temp1,theta1Temp2]*10**(-6), color = 'black', linewidth='1')

ax.set_title(r'$\theta=%d^{\circ}$, T = %d K' % ((theta* 180/math.pi), TVals[theta1Temp2]), y=1.15, x=0.63,pad=-14)

ax.set(xlabel = r'$\hbar\omega$ (meV)', ylabel=r'$f(\omega)$ (s$^{-1}$m$^{-2}$K$^{-1}$)')
ax.tick_params(which='both',direction='in')
ax.margins(0)
plt.savefig('GIntegrandWRTEnergyLocalUNSCREENEDdIndex=%d%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (dIndex, material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals), bbox_inches='tight')

# dLineCutPlot

array = [
    [1],
]

fig = pplt.figure(share=0, refwidth=2.5)#, refaspect = 1.333)
axs = fig.subplots(array)

axs.format(
    abc = False,
    grid= False
)

print(np.shape(GLineCutInd))
axs[0].plot(dLineCutVals, GLineCutInd*10**(-6), color = 'black')
axs[0].format(title = r'$\theta=%dº$' % theta, xlabel = r'$d$(Å)', ylabel = r'G $(\frac{MW}{m^2K})$', yscale='log', xscale='log')#, yscale='log')

plt.savefig('GLineCutIndLocalLogLogUNSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TVals[T1Index], TVals[T2Index], modStrengthFactor, dLineCutMin, dLineCutMax, numdLineCutVals), bbox_inches='tight')

"""
SCREENED
"""

@njit(cache=True)
def computeU(q, d):
    Vc = np.zeros((2*numgVals, 2*numgVals), dtype=np.complex128)
    for i in range(2*numgVals):
        g = gVals[i%numgVals]
        if np.linalg.norm(q+g) != 0:
            for j in range(2*numgVals):
                if i == j:
                    Vc[i][j] = (2*np.pi/(np.linalg.norm(q+g))) * eSquaredOvere #meV angstrom
                elif i%numgVals == j%numgVals:
                    Vc[i][j] = (2*np.pi/(np.linalg.norm(q+g))) * np.exp(-np.linalg.norm(q+g)*epsilonEffD*d) * eSquaredOvere #meV angstrom
    return(Vc)

@njit(cache=True)
def computeUsc(q, d, Pi1, Pi2):
    Vc = np.zeros((2*numgVals, 2*numgVals), dtype=np.complex128)
    for i in range(2*numgVals):
        g = gVals[i%numgVals]
        if np.linalg.norm(q+g) != 0:
            for j in range(2*numgVals):
                if i == j:
                    Vc[i][j] = (2*np.pi/(np.linalg.norm(q+g))) * eSquaredOvere #meV angstrom
                elif i%numgVals == j%numgVals:
                    Vc[i][j] = (2*np.pi/(np.linalg.norm(q+g))) * np.exp(-np.linalg.norm(q+g)*epsilonEffD*d) * eSquaredOvere #meV angstrom
    Pi = Pi1 + Pi2
    Vsc = np.linalg.inv(np.identity(2*numgVals)-Vc@Pi)@Vc
    return(Vsc)

@njit(cache=True) #njit only offers marginal benefits herefrom my testting. Eg less than one part on ten time
def computePMatrixIntegrandScreened():
    PMatrixIntegrand = np.zeros((numdVals, numTVals, numTVals, numhbarwVals), dtype=np.complex128)
    for qRedIndex in range(numRedMesh):
        print()
        print('qRedIndex:', qRedIndex)
        print()
        q, qIndex, offset = sendToMesh(reducedMesh[qRedIndex], mesh, numMesh, K, firstAndSecondShells)
        print('q:', q)
        if np.array_equal(q,  np.array([0,0])) is False:
            for hbarwIndex in range(numhbarwVals):
                #start = timeit.default_timer()
                #print('hbarwIndex:', hbarwIndex)
                hbarw = hbarwVals[hbarwIndex]
                for dIndex in range(numdVals):
                    d = dVals[dIndex]
                    #Usc = computeU(q, d) #THIS IS SET TO UNSCREENED RIGHT NOW FOR TESTING PURPOSES
                    #UscDagger = np.transpose(np.conj(Usc))
                    for ThIndex in range(numTVals):
                        Th = TVals[ThIndex]
                        for TcIndex in range(numTVals):
                            Tc = TVals[TcIndex]
                            DDRFh = np.zeros((2*numgVals, 2*numgVals), dtype=np.complex128)
                            DDRFh[0:numgVals, 0:numgVals] = megaMatrix[qRedIndex, hbarwIndex, ThIndex]
                            DDRFhAH = (DDRFh - np.transpose(np.conj(DDRFh)))/2
                            DDRFc = np.zeros((2*numgVals, 2*numgVals), dtype=np.complex128)
                            DDRFc[numgVals:2*numgVals, numgVals:2*numgVals] = megaMatrix[qRedIndex, hbarwIndex, TcIndex]
                            DDRFcAH = (DDRFc - np.transpose(np.conj(DDRFc)))/2
                            Usc = computeUsc(q, d, DDRFh, DDRFc)
                            UscDagger = np.transpose(np.conj(Usc))
                            trace = np.trace(DDRFhAH@Usc@DDRFcAH@UscDagger)
                            if hbarw != 0:
                                expTermHot = (np.exp(hbarw/(Kb*Th))-1)**(-1)
                                expTermCold = (np.exp(hbarw/(Kb*Tc))-1)**(-1)
                                PMatrixIntegrand[dIndex, ThIndex, TcIndex, hbarwIndex] += hbarw * trace * (expTermCold - expTermHot) * reducedMeshCounter[qRedIndex]
                #stop = timeit.default_timer()
                #print('time for hbaromega val:', (stop-start))
    PMatrixIntegrand = PMatrixIntegrand * 2 * (math.pi*hbar)**(-1) * (A*(((N**2)-1)/N**2))**(-1) * (SquareMetersPerSqareAngstrom)**(-1) * WattSecondsPermeV
    return(PMatrixIntegrand)

PMatrixIntegrandScreened = computePMatrixIntegrandScreened()

def PArrayGIntegrandWRTEnergyGArrayScreened(PMatrixIntegrand):
        dhbarw = hbarwVals[1]-hbarwVals[0]
        PArray = np.sum(PMatrixIntegrand, axis = 3)*dhbarw #(d,T1,T2)
        DeltaT = np.subtract.outer(TVals, TVals)
        ones = np.ones_like(DeltaT)
        OneOverDeltaT = np.divide(ones, DeltaT, out=np.zeros_like(ones), where=DeltaT!=0)
        GIntegrandWRTEnergy = np.einsum('li, klij -> klij', OneOverDeltaT, PMatrixIntegrand)
        GArray = -PArray*OneOverDeltaT
        for i in range(np.shape(GArray)[1]): #this function averages the diagonal
            if i==0:
                GArray[:,i,i] = GArray[:,i, i+1]
            if i==(np.shape(GArray)[1]-1):
                GArray[:,i,i] = GArray[:,i, i-1]
            else:
                GArray[:,i,i] = (GArray[:,i, i-1] + GArray[:,i, i+1])/2
        return(PArray, GIntegrandWRTEnergy, GArray)

PArrayScreened, GIntegrandWRTEnergyScreened, GArrayScreened = PArrayGIntegrandWRTEnergyGArrayScreened(PMatrixIntegrandScreened)
PArrayScreened=np.abs(np.real(PArrayScreened))
GIntegrandWRTEnergyScreened = np.abs(np.real(GIntegrandWRTEnergyScreened))
GArrayScreened = np.abs(np.real(GArrayScreened))
PArrayNameScreened = str('PArrayLocalSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals))
np.save(PArrayNameScreened, np.abs(np.real(PArrayScreened)))
GIntegrandWRTEnergyNameScreened = str('GIntegrandWRTEnergyLocalSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals))
np.save(GIntegrandWRTEnergyNameScreened, np.abs(np.real(GIntegrandWRTEnergyScreened)))
GArrayNameScreened = str('GArrayLocalSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals))
np.save(GArrayNameScreened, np.abs(np.real(GArrayScreened)))


#LineCutInD

T1Index, T2Index = np.unravel_index(np.abs(GArrayScreened[0]).argmax(), GArrayScreened[0].shape)
megaMatrixValScreened = megaMatrix[:,:,T2Index:T2Index+2]
TValsFordLineCut = np.array([TVals[T1Index], TVals[T2Index]])
GLineCutInd = np.zeros_like(dLineCutVals)
thermalFactor = computeThermalFactorArray(TValsFordLineCut)
PIntegrandScreened = computePMatrixIntegrandScreened(megaMatrixValScreened, dLineCutVals, TValsFordLineCut, thermalFactor)
PofdScreened, GIntegrandofdScreened, GofdScreened = PArrayGIntegrandWRTEnergyGArrayScreened(TValsFordLineCut, PIntegrandScreened)
GLineCutIndScreened = -np.real(GofdScreened[:,1,0])
GLineCutIndNameScreened = str('GLineCutIndLocalLogSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TVals[T1Index], TVals[T2Index], modStrengthFactor, dLineCutMin, dLineCutMax, numdLineCutVals))
np.save(GLineCutIndNameScreened, GLineCutIndScreened)

array = [
    [1],
]

fig = pplt.figure(share=0, refwidth=2.5)#, refaspect = 1.333)
axs = fig.subplots(array)

axs.format(
    abc = False,
    grid= False
)

axs[0].plot(dLineCutVals, GLineCutIndScreened*10**(-6), color = 'black')
axs[0].format(title = r'$\theta=%dº$' % theta, xlabel = r'$d$(Å)', ylabel = r'G $(\frac{MW}{m^2K})$', yscale='log', xscale='log')#, yscale='log')

plt.savefig('GLineCutIndLocalLogLogSCREENED%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TVals[T1Index], TVals[T2Index], modStrengthFactor, dLineCutMin, dLineCutMax, numdLineCutVals), bbox_inches='tight')

"""
PLOTS
"""

# G color plot

dIndex = 0

fig, ax = plt.subplots()

imshow1 = ax.imshow(GArrayScreened[dIndex]*10**(-6), cmap = 'hot', origin = 'lower', extent = [10,1000,10,1000])

cbar1 = plt.colorbar(imshow1)

cbar1.ax.set_xlabel(r'$G_{12}$ $(\frac{MW}{m^2K})$', loc='left')

ax.set_aspect('equal', adjustable='box')
ax.set_xticks([10,250,500,750,1000])
ax.set_yticks([10,250,500,750,1000])
ax.minorticks_off()

theta1Temp1, theta1Temp2 = np.unravel_index(np.abs(GArrayScreened[dIndex]).argmax(), GArrayScreened[dIndex].shape)

ax.set_title(r'$\theta=%d^{\circ}$' % (theta*180/math.pi), y=1.0, x=0.8,pad=-14, color='white')

ax.set(xlabel = r'T$_{1}$ (K)', ylabel=r'T$_{2}$ (K)')

plt.savefig('GArrayColorPlotLocalSCREENEDdIndex=%d%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (dIndex, material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals), bbox_inches='tight')

dIndex = 1

fig, ax = plt.subplots()

imshow1 = ax.imshow(GArrayScreened[dIndex]*10**(-6), cmap = 'hot', origin = 'lower', extent = [10,1000,10,1000])

cbar1 = plt.colorbar(imshow1)

cbar1.ax.set_xlabel(r'$G_{12}$ $(\frac{MW}{m^2K})$', loc='left')

ax.set_aspect('equal', adjustable='box')
ax.set_xticks([10,250,500,750,1000])
ax.set_yticks([10,250,500,750,1000])
ax.minorticks_off()

theta1Temp1, theta1Temp2 = np.unravel_index(np.abs(GArrayScreened[dIndex]).argmax(), GArrayScreened[dIndex].shape)

ax.set_title(r'$\theta=%d^{\circ}$' % (theta*180/math.pi) , y=1.0, x=0.8,pad=-14, color='white')

ax.set(xlabel = r'T$_{1}$ (K)', ylabel=r'T$_{2}$ (K)')

plt.savefig('GArrayColorPlotLocalSCREENEDdIndex=%d%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (dIndex, material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals), bbox_inches='tight')

# G integrand plot

dIndex = 0

fig, ax = plt.subplots()

theta1Temp1, theta1Temp2 = np.unravel_index(np.abs(GArrayScreened[dIndex]).argmax(), GArrayScreened[dIndex].shape)

ax.plot(hbarwVals, 6.242*10**(21)*GIntegrandWRTEnergyScreened[dIndex,:, theta1Temp1,theta1Temp2]*10**(-6), color = 'black', linewidth='1')

ax.set_title(r'$\theta=%d^{\circ}$, T = %d K' % ((theta* 180/math.pi), TVals[theta1Temp2]), y=1.15, x=0.63,pad=-14)

ax.set(xlabel = r'$\hbar\omega$ (meV)', ylabel=r'$f(\omega)$ (s$^{-1}$m$^{-2}$K$^{-1}$)')
ax.tick_params(which='both',direction='in')
ax.margins(0)
plt.savefig('GIntegrandWRTEnergyLocalSCREENEDdIndex=%d%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (dIndex, material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals), bbox_inches='tight')

dIndex = 1

fig, ax = plt.subplots()

theta1Temp1, theta1Temp2 = np.unravel_index(np.abs(GArray[dIndex]).argmax(), GArray[dIndex].shape)

ax.plot(hbarwVals, 6.242*10**(21)*GIntegrandWRTEnergyScreened[dIndex,:, theta1Temp1,theta1Temp2]*10**(-6), color = 'black', linewidth='1')

ax.set_title(r'$\theta=%d^{\circ}$, T = %d K' % ((theta* 180/math.pi), TVals[theta1Temp2]), y=1.15, x=0.63,pad=-14)

ax.set(xlabel = r'$\hbar\omega$ (meV)', ylabel=r'$f(\omega)$ (s$^{-1}$m$^{-2}$K$^{-1}$)')
ax.tick_params(which='both',direction='in')
ax.margins(0)
plt.savefig('GIntegrandWRTEnergyLocalSCREENEDdIndex=%d%s%sN=%dShells=%dHbareta=%.2fTheta=%snu=%denergy%d,%d,%dTVals=%d,%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (dIndex, material, date, N, numShells, hbareta, (theta*180/math.pi), nu, hbarwMin, hbarwMax, numhbarwVals, TMin,TMax, numTVals, modStrengthFactor, dMin, dMax, numdVals), bbox_inches='tight')
