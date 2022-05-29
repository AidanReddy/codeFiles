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
theta = 4 * math.pi/180 # twist angle
screened = True
material = 'MoS2' #WS2
hbareta = 1 # meV
hbarwMin = 0 # meV
hbarwMax = 200 * theta *180/math.pi #meV
numhbarwVals = 500
modStrengthFactor = 1 #multiplicative factor for moire potential
TVals = np.linspace(10,1000,100)
dLineCutMin = np.log10(0.1)
dLineCutMax = np.log10(40)
numdLineCutVals = 10
dLineCutVals = np.logspace(dLineCutMin,dLineCutMax,numdLineCutVals)
dMin = dLineCutMin
dMax = dLineCutMax
numdVals = numdLineCutVals
dVals = dLineCutVals
numdVals = np.shape(dVals)[0]
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

eSquaredOvere =  180951.2818 * 1/dielectricConstant #meV * angstrom
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
if theta == 1 * math.pi/180 :
    megaMatrix=np.load('megaMatrixLocalMoS2Mar312022N=7Shells=4Hbareta=1.00Theta=1.0energy0,300,500Ef=-14TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')
    if screened == False:
        GArray=np.load('GArrayLocalUNSCREENEDMoS2Apr032022N=7Shells=4Hbareta=1.00Theta=1.0energy0,200,500TVals=10,1000,100ModFactor=1dVals=0,20,2.npy')
    if screened == True:
        GArray=np.load('GArrayLocalSCREENEDMoS2Apr052022N=7Shells=4Hbareta=1.00Theta=1.0energy0,200,500TVals=10,990,50ModFactor=1dVals=0,20,2.npy')

if theta == 2 * math.pi/180 :
    megaMatrix=np.load('megaMatrixLocalMoS2Mar312022N=7Shells=4Hbareta=1.00Theta=2.0energy0,400,500Ef=-26TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')
    if screened == False:
        GArray = np.load('GArrayLocalUNSCREENEDMoS2Apr022022N=7Shells=4Hbareta=1.00Theta=2.0energy0,400,500TVals=10,1000,100ModFactor=1dVals=0,20,2.npy')
    if screened == True:
        GArray= np.load('GArrayLocalSCREENEDMoS2Apr022022N=7Shells=4Hbareta=1.00Theta=2.0energy0,400,500TVals=10,990,50ModFactor=1dVals=0,20,2.npy')

if theta == 3 * math.pi/180 :
    megaMatrix=np.load('megaMatrixLocalMoS2Mar242022N=7Shells=4Hbareta=1.00Theta=3.0energy0,800,500Ef=-49TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')
    if screened == False:
        GArray = np.abs(np.real(np.load('GArrayLocalUNSCREENEDMoS2Mar242022N=7Shells=4Hbareta=1.00Theta=3.0energy0,800,500TVals=10,1000,100ModFactor=1dVals=0,20,2.npy')))
    if screened == True:
        GArray=np.abs(np.load('GArrayLocalSCREENEDMoS2Apr032022N=7Shells=4Hbareta=1.00Theta=3.0energy0,600,500TVals=10,990,50ModFactor=1dVals=0,20,2.npy'))

if theta == 4 * math.pi/180 :
    megaMatrix=np.load('megaMatrixLocalMoS2Mar052022N=7Shells=4Hbareta=1.00Theta=4.0energy0,800,500Ef=-81TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')
    if screened == False:
        GArray = np.abs(np.real(np.load('GArrayLocalUNSCREENEDMoS2Apr032022N=7Shells=4Hbareta=1.00Theta=4.0energy0,800,500TVals=10,1000,100ModFactor=1dVals=0,20,2.npy')))
    if screened == True:
        GArray = np.load('GArrayLocalSCREENEDMoS2Apr022022N=7Shells=4Hbareta=1.00Theta=4.0energy0,800,500TVals=10,990,50ModFactor=1dVals=0,20,2.npy')

#megaMatrix = np.load('megaMatrixLocalMoS2Mar102022N=7Shells=4Hbareta=0.10Theta=1.0energy0,300,500Ef=-14TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')
#megaMatrix = np.load('megaMatrixLocalMoS2Mar092022N=7Shells=4Hbareta=0.20Theta=2.0energy0,300,500Ef=-26TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')
#megaMatrix = np.load('megaMatrixLocalMoS2Mar252022N=7Shells=4Hbareta=0.10Theta=3.0energy0,800,500Ef=-49TVals=10,1000,100ModFactor=1Epsilon=5.104586173236768.npy')

if screened == True:
    megaMatrixSparse = megaMatrix[:,:,::2,:,:] #takes in a megamatrix that has more T values for unscreened calculation and de-densifies it to be able to do a screened calculation
    del megaMatrix
    megaMatrix = megaMatrixSparse
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
mesh, reducedMesh, reducedMeshCounter, meshToReducedMeshIndexMap = cmb.computeMesh(N, am)
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
    UDiagonal = (1/2)*eSquaredOvere*np.divide(ones, gqDiagonalTimesqPlusGNormdVals, out=np.zeros_like(ones), where=gqDiagonalTimesqPlusGNormdVals!=0) #when dividing the two matrcies, set elements that are divides by zero to zero
    UOffDiagonal=exponentialDecayFactor*UDiagonal
    U = np.block([[UDiagonal, UOffDiagonal], [UOffDiagonal, UDiagonal]]) #(d, q, g, gPrime)
    return(U)

@njit(cache=True)
def computeUsc(q, d, Pi1, Pi2):
    Vc = np.zeros((2*numgVals, 2*numgVals), dtype=np.complex128)
    for i in range(2*numgVals):
        g = gVals[i%numgVals]
        if np.linalg.norm(q+g) != 0:
            for j in range(2*numgVals):
                if i == j:
                    Vc[i][j] = (1/(2*np.linalg.norm(q+g))) * eSquaredOvere #meV angstrom
                elif i%numgVals == j%numgVals:
                    Vc[i][j] = (1/(2*np.linalg.norm(q+g))) * np.exp(-np.linalg.norm(q+g)*epsilonEffD*d) * eSquaredOvere #meV angstrom
    Pi = Pi1 + Pi2
    Vsc = np.linalg.inv(np.identity(2*numgVals)-Vc@Pi)@Vc
    return(Vsc)

def computeThermalFactorArray(TVals):
    betaVals = 1/(Kb*TVals)
    numTVals=np.shape(TVals)[0]
    hbarwBroadcast = np.transpose(np.tile(hbarwVals, (numTVals, 1)))
    print(np.shape(hbarwBroadcast))
    print(np.shape(betaVals))
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

def computePMatrixIntegrandVectorized(megaMatrix, dVals, TVals, thermalFactor):
    print('pre U')
    U = constructU(megaMatrix, dVals, TVals)
    #thermalFactor = computeThermalFactorArray()
    print('post U')
    AHMegaMatrix = (megaMatrix - np.conj(np.einsum('...ij->...ji', megaMatrix)))/2 #(q,w,T,g,gPrime)
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

@njit(cache=True) #njit only offers marginal benefits herefrom my testting. Eg less than one part on ten time
def computePMatrixIntegrandScreened(megaMatrix, dVals, TVals, thermalFactor):
    numTVals = np.shape(TVals)[0]
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

def PArrayGIntegrandWRTEnergyGArray(TVals, PMatrixIntegrand):
    dhbarw = hbarwVals[1]-hbarwVals[0]
    PMatrixIntegratedOverQ = np.einsum('ijlkm, j -> ilkm', PMatrixIntegrand, reducedMeshCounter)
    PArray = np.sum(PMatrixIntegratedOverQ, axis = 1)*dhbarw #(d,T1,T2)
    DeltaT = np.subtract.outer(TVals, TVals)
    GIntegrandWRTEnergy = np.einsum('ij, klij -> klij', DeltaT, PMatrixIntegratedOverQ)
    ones = np.ones_like(DeltaT)
    OneOverDeltaT = np.divide(ones, DeltaT, out=np.zeros_like(ones), where=DeltaT!=0)
    GArray = -PArray*OneOverDeltaT
    for i in range(np.shape(GArray)[1]): #this function averages the diagonal
        if i==0:
            GArray[:,i,i] = GArray[:,i, i+1]
        if i==(np.shape(GArray)[1]-1):
            GArray[:,i,i] = GArray[:,i, i-1]
        else:
            GArray[:,i,i] = (GArray[:,i, i-1] + GArray[:,i, i+1])/2
    return(PArray, GIntegrandWRTEnergy, GArray)

def PArrayGIntegrandWRTEnergyGArrayScreened(TVals, PMatrixIntegrand):
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

#LineCutInD

T1Index, T2Index = np.unravel_index(np.abs(GArray[0]).argmax(), GArray[0].shape)

print('T1Index, T2Index:', T1Index, T2Index)

print(np.shape(megaMatrix))
megaMatrixVal = megaMatrix[:,:,T2Index:T2Index+2]

print(np.shape(megaMatrixVal))

TValsFordLineCut = np.array([TVals[T1Index], TVals[T2Index]])

print('TValsFordLineCut:', TValsFordLineCut)

GLineCutInd = np.zeros_like(dLineCutVals)

thermalFactor = computeThermalFactorArray(TValsFordLineCut)

if screened == False:
    PIntegrand = computePMatrixIntegrandVectorized(megaMatrixVal, dLineCutVals, TValsFordLineCut, thermalFactor)
    Pofd, GIntegrandofd, Gofd = PArrayGIntegrandWRTEnergyGArray(TValsFordLineCut, PIntegrand)
if screened == True:
    PIntegrand = computePMatrixIntegrandScreened(megaMatrixVal, dLineCutVals, TValsFordLineCut, thermalFactor)
    Pofd, GIntegrandofd, Gofd = PArrayGIntegrandWRTEnergyGArrayScreened(TValsFordLineCut, PIntegrand)

print(np.shape(PIntegrand))
print(np.shape(Gofd))
GLineCutInd = -np.real(Gofd[:,1,0])
print(GLineCutInd)
GLineCutIndName = str('GLineCutIndLocalLog%sScreened%s%sN=%dShells=%dHbareta=%.2fTheta=%senergy%d,%d,%dTVals=%d,%dModFactor=%sdVals=%d,%d,%d' % (screened, material, date, N, numShells, hbareta, (theta*180/math.pi), hbarwMin, hbarwMax, numhbarwVals, TVals[T1Index], TVals[T2Index], modStrengthFactor, dLineCutMin, dLineCutMax, numdLineCutVals))
np.save(GLineCutIndName, GLineCutInd)

array = [
    [1],
]

fig = pplt.figure(share=0, refwidth=2.5)#, refaspect = 1.333)
axs = fig.subplots(array)

axs.format(
    abc = False,
    grid= False,
    titleloc='ur',
)

thetadeg = np.int(theta*(180/np.pi))

print(np.shape(GLineCutInd))
axs[0].plot(dLineCutVals, GLineCutInd*10**(-6), color = 'black')
axs[0].format(title = r'$\theta=%dº$' % thetadeg, xlabel = r'$d$(Å)', ylabel = r'G $(\frac{MW}{m^2K})$', yscale='log')
plt.savefig('GLineCutIndLocalLog%sScreened%s%sN=%dShells=%dHbareta=%.2fTheta=%senergy%d,%d,%dTVals=%d,%dModFactor=%sdVals=%d,%d,%d.pdf' % (screened, material,date, N, numShells, hbareta, (theta*180/math.pi), hbarwMin, hbarwMax, numhbarwVals, TVals[T1Index], TVals[T2Index], modStrengthFactor, dLineCutMin, dLineCutMax, numdLineCutVals), bbox_inches='tight')

plt.show()
