import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from numba import njit
import timeit
from datetime import date

today = date.today()
date = today.strftime("%b%d%Y")

import EDandMatrixElementsModule as ed
import continuumModelBandsModule as cmb
import materialContinuumModelParameters as cmp

dataSaveDir='/Users/aidanreddy/Desktop/resonantEnergyTransfer/data/'

"""
How to use timeit:
start = timeit.default_timer()
x
stop = timeit.default_timer()
print('time for x: ', stop - start)
"""

#np.shape(basisStatesMatrix)[1] #5 # cuts off number of states to sum over in Fermi's golden rule
#GLineCutInTofdandsInteractingandSingleParticle=np.load('GLineCutInTofdandsInteractingandSingleParticleApr072022theta=1EpsilonED=5dMin=0dMax=20numdVals=2sMin=0sMax=0numsVals=1N=7Included=15TMin=30TMax=1000numTVals=100Gamma0=1.npy')
#GLineCutInTofdandsInteractingandSingleParticle=np.load('GLineCutInTofdandsInteractingandSingleParticleApr102022theta=2Mod=1EpsilonED=5dMin=0dMax=20numdVals=2sMin=0sMax=0numsVals=1N=7Included=15TMin=30TMax=1000numTVals=100Gamma0=1.npy')
#GLineCutInTofdandsInteractingandSingleParticle=np.load('GLineCutInTofdandsInteractingandSingleParticleApr102022theta=3.02Mod=1EpsilonED=5dMin=0dMax=20numdVals=2sMin=0sMax=0numsVals=1N=7Included=15TMin=30TMax=1000numTVals=100Gamma0=1.npy')
#GLineCutInTofdandsInteractingandSingleParticle=np.load('GLineCutInTofdandsInteractingandSingleParticleApr102022theta=4Mod=1EpsilonED=5dMin=0dMax=20numdVals=2sMin=0sMax=0numsVals=1N=7Included=15TMin=30TMax=1000numTVals=100Gamma0=1.npy')

Kb = 0.08617 # meV per Kelvin
hbar = 6.582 * 10**(-13) # meV * s
JoulesPermeV = 1.602 * 10**(-22)
SquareMetersPerSqareAngstrom = 10**(-20)
electronMass = 5.685 * 10**(-29) # meV *(second/Å)^2
eSquaredOvere0 =  14400 #meV * angstrom

def computeParameters(theta, a, V1, V2, V3, mStar, dielectricConstant, modStrengthFactor): 
    aM=a*(180/np.pi)/theta # angstroms
    gamma = 8*np.pi**2*(V1-6*V2+4*V3)* modStrengthFactor # meV. confirmed that this is the correct form of the harmonic approximation 04/25/2022.
    hbaromega = hbar*np.sqrt(gamma/(mStar*electronMass))/aM # sqrt(meV/electron mass)/angstrom
    l = np.sqrt(hbar**2/((mStar*electronMass)*hbaromega))
    coulombEnergy = eSquaredOvere0/(dielectricConstant*l) # e^2 = 1.44 eV*nm
    ratio = coulombEnergy/hbaromega
    return(aM, gamma, hbaromega, l, coulombEnergy, ratio)

def calculateGammaofN(theta, material, numGammaNContinuumModel, modStrengthFactor, numShellsForGammaofN):
    cmb.theta = theta * np.pi/180 # twist angle
    cmb.N = 7 #7 ### Creates NxN k space mesh. N MUST BE ODD!!!!!
    cmb.numShells = numShellsForGammaofN # number of reciprocal lattice vector shells
    cmb.modStrengthFactor = modStrengthFactor #multiticative factor for moire potential
    cmb.electronMass = 5.856301 * 10**(-29) # meV *(second/Å)
    cmb.a, cmb.V1, cmb.V2, cmb.V3, cmb.phi, cmb.mStar = cmp.materialContinuumModelParameters(material)
    cmb.am = cmb.a/cmb.theta #moiré period in linear approximation
    cmb.A = (cmb.am**2) * np.sqrt(3)/2 * cmb.N**2 # define total lattice area A
    cmb.b1 = (4*np.pi/np.sqrt(3)) * (1/cmb.am) * np.array([1,0]) # define reciprocal basis vectors b1 and b2 for moire lattice
    cmb.b2 = (4*np.pi/np.sqrt(3)) * (1/cmb.am) * np.array([0.5, np.sqrt(3)/2])
    cmb.a1 = cmb.am * np.array([np.sqrt(3)/2, -1/2]) # define real basis vectors a1 and a2 for moire lattice
    cmb.a2 = cmb.am * np.array([0, 1])
    cmb.K = np.array([0,1])*(np.linalg.norm(cmb.b1)/np.sqrt(3))#((b1+b2)/np.linalg.norm(b1+b2)) * (np.linalg.norm(b1)/np.sqrt(3))
    cmb.shells = cmb.computeShell(cmb.numShells, cmb.am)
    cmb.gVals = cmb.shells#computeShell(np.floor(numShells/2))
    cmb.numgVals = np.shape(cmb.gVals)[0]
    cmb.numBands = np.shape(cmb.shells)[0]
    cmb.mesh, cmb.reducedMesh, cmb.reducedMeshCounter, cmb.meshToReducedMeshIndexMap = cmb.computeMesh(cmb.N, cmb.am)
    cmb.numMesh = np.shape(cmb.mesh)[0]
    cmb.numRedMesh = np.shape(cmb.reducedMesh)[0]
    megaEigValArray, megaEigVecArray = cmb.computeMegaEigStuff()
    megaEigValArray -= np.max(megaEigValArray)
    GammaofNContinuumModel = np.zeros(numGammaNContinuumModel)
    count = 0
    for NIndex in range(numGammaNContinuumModel):
        Ndegen = 2*(NIndex+1)
        bandSet = megaEigValArray[:,np.shape(megaEigValArray)[1]-count-Ndegen:(np.shape(megaEigValArray)[1]-count)]
        GammaofNContinuumModel[NIndex] += np.sqrt(np.var(bandSet))
        count += Ndegen
    return(GammaofNContinuumModel)

@njit
def computePartitionFunction(nu, beta, mu, Es_s, Es_a, EOneBody=np.array([0])): #dummy argument for EOneBody because I only need it for nu=2 and not for nu=4
    if nu==4:
        Z=0
        for energyVal_s in Es_s:
            Z+=np.exp(-(energyVal_s)*beta)
        for energyVal_a in Es_a:
            Z+=3*np.exp(-(energyVal_a)*beta) # factor of 3 because of triplet degeneracy
    if nu==2:
        Z=0
        #recall that the grand partition function is sum_i exp(-(Ei-Ni*mu)\beta)
        # two hole fock space
        for energyVal_s in Es_s:
            Z+=np.exp(-(energyVal_s-2*mu)*beta)
        for energyVal_a in Es_a:
            Z+=3*np.exp(-(energyVal_a-2*mu)*beta) # factor of 3 because of triplet degeneracy
        #one hole fock space
        for energyValOneBody in EOneBody:
            Z+=2*np.exp(-(energyValOneBody-1*mu)*beta) # factor of 2 for spin up and down degeneracy
        # zero hole fock space
        Z+=1 #np.exp(-(0-mu*0)*beta)
    return(Z)

@njit
def computeDetailedBalance(eStateEnergy, eStatePrimeEnergy, beta1, beta2, Z1, Z2):
    detailedBalance = ((2*Z1*Z2)**(-1))*(np.exp(-eStateEnergy*beta1)*np.exp(-eStatePrimeEnergy*beta2)- np.exp(-eStateEnergy*beta2)*np.exp(-eStatePrimeEnergy*beta1))
    return(detailedBalance)

@njit
def MuTtoN(mu, nu, T, energyValsFock1, energyValsFock2, Es_s, Es_a, EOneBody):
    n = 0
    #print('T:', T)
    beta = 1/(Kb*T)
    Z = computePartitionFunction(nu, beta, mu, Es_s, Es_a, EOneBody)
    for energyValFock1 in energyValsFock1:
        n += 1*np.exp(-(energyValFock1-1*mu)*beta)/Z
    for energyValFock2 in energyValsFock2:
        n += 2*np.exp(-(energyValFock2-2*mu)*beta)/Z
    return(n)

@njit
def computeMuOfTVector(TVals, hbaromega, n, nu, muOfTPrecision, EOneBody, Es_s, Es_a):
    numTVals = np.shape(TVals)[0] #note that n is <number of holes per moiré atom>
    energyValsFock1 = EOneBody
    energyValsFock2 = np.sort(np.concatenate((Es_s, Es_a)))
    MuOfTVector = np.zeros(numTVals)
    trialMu = 3*hbaromega
    for TIndex, T in np.ndenumerate(TVals):
        #print('TIndex:', TIndex)
        nFound = False
        trialn=100 #dummy value whose only important property is being > 1
        while nFound == False:
            if trialn < (1+muOfTPrecision)*n:
                print('error in mu of t calculation!')
            #print('trialMu/hbaromega:', trialMu/hbaromega)
            trialMu -= 0.01*muOfTPrecision*hbaromega # be weary! If i make this step too big, it wont work. e.g. 0.01 wasnt working and it took me a while to get to the bottom of it. But now I got it! So, if this function is causing errors, try making the mu step smaller.
            trialn = MuTtoN(trialMu, nu, T, energyValsFock1, energyValsFock2, Es_s, Es_a, EOneBody)
            #print('trialn:', trialn)
            nFound = trialn < (1+muOfTPrecision)*n and trialn > (1-muOfTPrecision)*n
        MuOfTVector[TIndex] = trialMu
    return(MuOfTVector)

@njit
def computeLorentzianFactor(eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, Es, eigenstateIndextoNArray, GammaofN, Gamma0):
    NAi = eigenstateIndextoNArray[eigenstateAiIndex]
    NAj = eigenstateIndextoNArray[eigenstateAjIndex]
    NBi = eigenstateIndextoNArray[eigenstateBiIndex]
    NBj = eigenstateIndextoNArray[eigenstateBjIndex]
    detuning = abs(Es[eigenstateAiIndex]-Es[eigenstateAjIndex]+(Es[eigenstateBiIndex]-Es[eigenstateBjIndex]))
    GammaAi = GammaofN[NAi] #note: this definition od GammaN gets wishy-washy when bands start crossing
    GammaAj = GammaofN[NAj]
    GammaBi = GammaofN[NBi]
    GammaBj = GammaofN[NBj]
    Gamma = 4*Gamma0 #GammaAi+GammaAj+GammaBi+GammaBj+4*Gamma0
    lorentzianFactor = (1/np.pi)*(Gamma/2)/((detuning)**2+(Gamma/2)**2)
    return(lorentzianFactor)

def computeEnergyTransferTLineCutSingleParticle(hbaromega, T1T2Vals,beta1beta2Vals, Z1Z2Vals, dTilde, sTilde, mStar, epsForEnergyTransfer, numEStatesIncluded, A, ESingleParticle_s, ESingleParticle_a, basisStatesMatrix_s, basisStatesMatrix_a, GammaofN, Gamma0):
    QDot = np.zeros(np.shape(T1T2Vals)[0])
    numStatesIncludedTot = 2*numEStatesIncluded # numEStatesIncluded is defined per singlet/triplet
    EVals = np.vstack((ESingleParticle_s, ESingleParticle_a))
    VSquaredArray = np.zeros((numStatesIncludedTot, numStatesIncludedTot, numStatesIncludedTot, numStatesIncludedTot))
    statesMatrix = np.hstack(basisStatesMatrix_s, basisStatesMatrix_a)
    for basisStateAiIndex in range(numStates):
        basisStateAi = statesMatrix[:, basisStateAiIndex]
        basisStateAiEnergy = ESingleParticle_s[basisStateAiIndex]
        nRpAi=statesMatrix[0]
        nRmAi=statesMatrix[1]
        nrpAi=statesMatrix[2]
        nrmAi=statesMatrix[3]
        basisStateAi_a = basisStatesMatrix_a[:, basisStateAiIndex]
        basisStateAiEnergy_a = ESingleParticle_a[basisStateAiIndex]
        nRpAi_a=basisStateAi_a[0]
        nRmAi_a=basisStateAi_a[1]
        nrpAi_a=basisStateAi_a[2]
        nrmAi_a=basisStateAi_a[3]
        for basisStateAjIndex in range(numEStatesIncluded):
            basisStateAj_s = basisStatesMatrix_s[:, basisStateAjIndex]
            basisStateAjEnergy_s = ESingleParticle_s[basisStateAjIndex]
            nRpAj_s=basisStateAj_s[0]
            nRmAj_s=basisStateAj_s[1]
            nrpAj_s=basisStateAj_s[2]
            nrmAj_s=basisStateAj_s[3]
            basisStateAj_a = basisStatesMatrix_a[:, basisStateAjIndex]
            basisStateAjEnergy_a = ESingleParticle_a[basisStateAjIndex]
            nRpAj_a=basisStateAj_a[0]
            nRmAj_a=basisStateAj_a[1]
            nrpAj_a=basisStateAj_a[2]
            nrmAj_a=basisStateAj_a[3]
            for basisStateBiIndex in range(numEStatesIncluded):
                basisStateBi_s = basisStatesMatrix_s[:, basisStateBiIndex]
                basisStateBiEnergy_s = ESingleParticle_s[basisStateBiIndex]
                nRpBi_s=basisStateBi_s[0]
                nRmBi_s=basisStateBi_s[1]
                nrpBi_s=basisStateBi_s[2]
                nrmBi_s=basisStateBi_s[3]
                basisStateBi_a = basisStatesMatrix_a[:, basisStateBiIndex]
                basisStateBiEnergy_a = ESingleParticle_a[basisStateBiIndex]
                nRpBi_a=basisStateBi_a[0]
                nRmBi_a=basisStateBi_a[1]
                nrpBi_a=basisStateBi_a[2]
                nrmBi_a=basisStateBi_a[3]
                for basisStateBjIndex in range(numEStatesIncluded):
                    basisStateBj_s = basisStatesMatrix_s[:, basisStateBjIndex]
                    basisStateBjEnergy_s = ESingleParticle_s[basisStateBjIndex]
                    nRpBj_s=basisStateBj_s[0]
                    nRmBj_s=basisStateBj_s[1]
                    nrpBj_s=basisStateBj_s[2]
                    nrmBj_s=basisStateBj_s[3]
                    basisStateBj_a = basisStatesMatrix_a[:, basisStateBjIndex]
                    basisStateBjEnergy_a = ESingleParticle_a[basisStateBjIndex]
                    nRpBj_a=basisStateBj_a[0]
                    nRmBj_a=basisStateBj_a[1]
                    nrpBj_a=basisStateBj_a[2]
                    nrmBj_a=basisStateBj_a[3]
                    V_s=ed.Coul_hh_4body_generalSeparation(dTilde,sTilde,hbaromega,nRpAi_s,nRmAi_s,nrpAi_s,nrmAi_s,nRpAj_s,nRmAj_s,nrpAj_s,nrmAj_s,nRpBi_s,nRmBi_s,nrpBi_s,nrmBi_s,nRpBj_s,nRmBj_s,nrpBj_s,nrmBj_s,mStar,epsForEnergyTransfer)
                    VSquaredArray_s[basisStateAiIndex, basisStateAjIndex, basisStateBiIndex, basisStateBjIndex]= abs(V_s)**2
                    V_a=ed.Coul_hh_4body_generalSeparation(dTilde,sTilde,hbaromega,nRpAi_a,nRmAi_a,nrpAi_a,nrmAi_a,nRpAj_a,nRmAj_a,nrpAj_a,nrmAj_a,nRpBi_a,nRmBi_a,nrpBi_a,nrmBi_a,nRpBj_a,nRmBj_a,nrpBj_a,nrmBj_a,mStar,epsForEnergyTransfer)
                    VSquaredArray_a[basisStateAiIndex, basisStateAjIndex, basisStateBiIndex, basisStateBjIndex]= abs(V_a)**2
    for basisStateAiIndex in range(numEStatesIncluded):
        basisStateAi_s = basisStatesMatrix_s[:, basisStateAiIndex]
        basisStateAiEnergy_s = ESingleParticle_s[basisStateAiIndex]
        NAi_s = int(1.01*basisStateAiEnergy_s/hbaromega-2)
        basisStateAi_a = basisStatesMatrix_a[:, basisStateAiIndex]
        basisStateAiEnergy_a = ESingleParticle_a[basisStateAiIndex]
        NAi_a = int(1.01*basisStateAiEnergy_a/hbaromega-2)
        for basisStateAjIndex in range(numEStatesIncluded):
            basisStateAj_s = basisStatesMatrix_s[:, basisStateAjIndex]
            basisStateAjEnergy_s = ESingleParticle_s[basisStateAjIndex]
            NAj_s = int(1.01*basisStateAjEnergy_s/hbaromega-2)
            basisStateAj_a = basisStatesMatrix_a[:, basisStateAjIndex]
            basisStateAjEnergy_a = ESingleParticle_a[basisStateAjIndex]
            NAj_a = int(1.01*basisStateAjEnergy_a/hbaromega-2)
            for basisStateBiIndex in range(numEStatesIncluded):
                basisStateBi_s = basisStatesMatrix_s[:, basisStateBiIndex]
                basisStateBiEnergy_s = ESingleParticle_s[basisStateBiIndex]
                NBi_s = int(1.01*basisStateBiEnergy_s/hbaromega-2)
                basisStateBi_a = basisStatesMatrix_a[:, basisStateBiIndex]
                basisStateBiEnergy_a = ESingleParticle_a[basisStateBiIndex]
                NBi_a = int(1.01*basisStateBiEnergy_a/hbaromega-2)
                for basisStateBjIndex in range(numEStatesIncluded):
                    basisStateBj_s = basisStatesMatrix_s[:, basisStateBjIndex]
                    basisStateBjEnergy_s = ESingleParticle_s[basisStateBjIndex]
                    NBj_s = int(1.01*basisStateBjEnergy_s/hbaromega-2)
                    basisStateBj_a = basisStatesMatrix_a[:, basisStateBjIndex]
                    basisStateBjEnergy_a = ESingleParticle_a[basisStateBjIndex]
                    NBj_a = int(1.01*basisStateBjEnergy_a/hbaromega-2)
                    VSquared_s = VSquaredArray_s[basisStateAiIndex,basisStateAjIndex,basisStateBiIndex,basisStateBjIndex]
                    VSquared_a = VSquaredArray_a[basisStateAiIndex,basisStateAjIndex,basisStateBiIndex,basisStateBjIndex]
                    for TIndex in range(np.shape(T1T2Vals)[0]):
                        beta1, beta2 = beta1beta2Vals[TIndex]
                        Z1, Z2 = Z1Z2Vals[TIndex]
                        detailedBalance_s = computeDetailedBalance(basisStateAiEnergy_s, basisStateBiEnergy_s, beta1, beta2, Z1, Z2)
                        detailedBalance_a = computeDetailedBalance(basisStateAiEnergy_a, basisStateBiEnergy_a, beta1, beta2, Z1, Z2)
                        deltaEnergy_s = ((basisStateAiEnergy_s-basisStateAjEnergy_s)-(basisStateBiEnergy_s-basisStateBjEnergy_s))/2
                        deltaEnergy_a = ((basisStateAiEnergy_a-basisStateAjEnergy_a)-(basisStateBiEnergy_a-basisStateBjEnergy_a))/2
                        detuning_s = abs((basisStateAiEnergy_s-basisStateAjEnergy_s)+(basisStateBiEnergy_s-basisStateBjEnergy_s))
                        detuning_a = abs((basisStateAiEnergy_a-basisStateAjEnergy_a)+(basisStateBiEnergy_a-basisStateBjEnergy_a))
                        GammaAi_s = GammaofN[NAi_s] #doesnt work once bands start crossing, need to revise
                        GammaAj_s = GammaofN[NAj_s]
                        GammaBi_s = GammaofN[NBi_s]
                        GammaBj_s = GammaofN[NBj_s] 
                        Gamma_s = GammaAi_s+GammaAj_s+GammaBi_s+GammaBj_s+4*Gamma0
                        GammaAi_a = GammaofN[NAi_a] #doesnt work once bands start crossing, need to revise
                        GammaAj_a = GammaofN[NAj_a] 
                        GammaBi_a = GammaofN[NBi_a]
                        GammaBj_a = GammaofN[NBj_a] 
                        Gamma_a = GammaAi_a+GammaAj_a+GammaBi_a+GammaBj_a+4*Gamma0
                        lorentzianFactor_s = (1/np.pi)*(Gamma_s/2)/((detuning_s)**2+(Gamma_s/2)**2)
                        lorentzianFactor_a = (1/np.pi)*(Gamma_a/2)/((detuning_a)**2+(Gamma_a/2)**2)
                        QDot_s[TIndex] += VSquared_s * detailedBalance_s * deltaEnergy_s * lorentzianFactor_s
                        QDot_a[TIndex] += VSquared_a * detailedBalance_a * deltaEnergy_a * lorentzianFactor_a
    QDot_s *= 2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2 #(last factor of two is because two moire atoms per unit cell) # outputs in Watts
    QDot_a *= 2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2
    QDot = QDot_s+QDot_a
    G_s = np.einsum('i,i-> i', abs(QDot_s), abs((T1T2Vals[:,0]-T1T2Vals[:,1])**(-1)))
    G_a = np.einsum('i,i-> i', abs(QDot_a), abs((T1T2Vals[:,0]-T1T2Vals[:,1])**(-1)))
    G = G_s + G_a
    return(QDot, G, G_s, G_a)

def computeEnergyTransferTLineCut(hbaromega, T1T2Vals, beta1beta2Vals, Z1Z2Vals, dTilde, sTilde, mStar, epsForEnergyTransfer, numEStatesIncluded, cutoff, A, Es, evecs_s, evecs_a, parityList, basisStates_s, basisStates_a, GammaofN, Gamma0, eigenstateIndextoNArray):
    # first, only include lowest numEStatesIncluded states (note that Es and evecs are already sorted)
    QDot = np.zeros(np.shape(T1T2Vals)[0])
    for eigenstateAiIndex in range(numEStatesIncluded):
        eigenStateAiEnergy = Es[eigenstateAiIndex]
        AiParity = parityList[eigenstateAiIndex]
        for eigenstateAjIndex in range(numEStatesIncluded):
            eigenStateAjEnergy = Es[eigenstateAjIndex]
            AjParity = parityList[eigenstateAjIndex]
            if AiParity == AjParity: # initial and final states of a given atom must have same parity (and, more restrictively, the same exact spin state)
                for eigenstateBiIndex in range(numEStatesIncluded):
                    eigenStateBiEnergy = Es[eigenstateBiIndex]
                    BiParity = parityList[eigenstateBiIndex]
                    for eigenstateBjIndex in range(numEStatesIncluded):
                        eigenStateBjEnergy = Es[eigenstateBjIndex]
                        BjParity = parityList[eigenstateBjIndex]
                        deltaEnergy = ((eigenStateAjEnergy-eigenStateAiEnergy)+(eigenStateBiEnergy-eigenStateBjEnergy)) # (eigenStateAjEnergy-eigenStateBjEnergy) #
                        if BiParity == BjParity:
                            lorentzianFactor = computeLorentzianFactor(eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, Es, eigenstateIndextoNArray, GammaofN, Gamma0)
                            V = ed.Coul_4body_estateestate(dTilde, sTilde, eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, cutoff, hbaromega, parityList, basisStates_s, basisStates_a, evecs_s, evecs_a, mStar, epsForEnergyTransfer)
                            VSquared = np.abs(V)**2
                            spinDegeneracyFactor = (3)**(AjParity+BjParity)
                            val = VSquared * deltaEnergy * lorentzianFactor * spinDegeneracyFactor
                            for TIndex in range(np.shape(T1T2Vals)[0]):
                                beta1, beta2 = beta1beta2Vals[TIndex]
                                Z1, Z2 = Z1Z2Vals[TIndex]
                                detailedBalance = computeDetailedBalance(eigenStateAjEnergy, eigenStateBjEnergy, beta1, beta2, Z1, Z2)
                                QDot[TIndex] += val*detailedBalance
    QDot *= 2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2 #(last factor of two is because two moire atoms per unit cell) # outputs in Watts
    G = np.einsum('i,i-> i', QDot, (T1T2Vals[:,0]-T1T2Vals[:,1])**(-1))
    return(QDot, G)

"""
thing = VSquared * detailedBalance * deltaEnergy * lorentzianFactor * spinDegeneracyFactor
if (thing*2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2*10**(-6)) > 1:
    print(thing*2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2*10**(-6))
    print('V/hbaromega:', np.abs(V)/hbaromega)
    print('detailedBalance:', detailedBalance)
    print('deltaEnergy/hbaromega:', deltaEnergy/hbaromega)
    print('lorentzianFactor:', lorentzianFactor)
    print('spinDegeneracyFactor:', spinDegeneracyFactor)
    print()

#if detailedBalance * (eigenStateAjEnergy-eigenStateBjEnergy) < -(1**(-10)):
#    print('hit!')
#    print(detailedBalance * (eigenStateAjEnergy-eigenStateBjEnergy))
#    print()

if val*2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2 < -1*10**(6):
        print('hit!')
        print(deltaEnergy/hbaromega)
        print('energy lost by A:', (eigenStateAjEnergy-eigenStateAiEnergy)/hbaromega)
        print('energy gained by B:', (eigenStateBiEnergy-eigenStateBjEnergy)/hbaromega)
        print('V/hbaromega:', np.abs(V)/hbaromega)
        print('detailedBalance:', detailedBalance)
        print('deltaEnergy/hbaromega:', deltaEnergy/hbaromega)
        print('lorentzianFactor:', lorentzianFactor)
        print('spinDegeneracyFactor:', spinDegeneracyFactor)
        print()
"""
def computeEnergyTransferTLineCutNu2(muOfTVector1, muOfTVector2, hbaromega, T1T2Vals,beta1beta2Vals, Z1Z2Vals, dTilde, mStar, epsForEnergyTransfer, numEStatesIncluded, cutoff, A, EOneBody, Es_s, Es_a, evecs_s, evecs_a, oneBodyStatesMatrix, basisStatesMatrix_s, basisStatesMatrix_a, GammaofN, Gamma0, eigenstateIndextoNArraySym, eigenstateIndextoNArrayAsym):
    sTilde = 0 #I can go back and generalize if I want
    #fock 2 subspace(interacting)
    QDotFock2 = np.zeros(np.shape(T1T2Vals)[0])
    VSquaredArray_s = np.zeros((numEStatesIncluded, numEStatesIncluded, numEStatesIncluded, numEStatesIncluded))
    VSquaredArray_a = np.zeros((numEStatesIncluded, numEStatesIncluded, numEStatesIncluded, numEStatesIncluded))
    for eigenstateAiIndex in range(numEStatesIncluded):
        eigenStateAiEnergy_s = Es_s[eigenstateAiIndex]
        eigenStateAiEnergy_a = Es_a[eigenstateAiIndex]
        for eigenstateAjIndex in range(numEStatesIncluded):
            eigenStateAjEnergy_s = Es_s[eigenstateAjIndex]
            eigenStateAjEnergy_a = Es_a[eigenstateAjIndex]
            for eigenstateBiIndex in range(numEStatesIncluded):
                eigenStateBiEnergy_s = Es_s[eigenstateBiIndex]
                eigenStateBiEnergy_a = Es_a[eigenstateBiIndex]
                for eigenstateBjIndex in range(numEStatesIncluded):
                    eigenStateBjEnergy_s = Es_s[eigenstateBjIndex]
                    eigenStateBjEnergy_a = Es_a[eigenstateBjIndex]
                    V_s = ed.Coul_4body_estateestate(dTilde, sTilde, eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, cutoff, hbaromega, basisStatesMatrix_s, evecs_s, mStar, epsForEnergyTransfer)
                    V_a = ed.Coul_4body_estateestate(dTilde, sTilde, eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, cutoff, hbaromega, basisStatesMatrix_a, evecs_a, mStar, epsForEnergyTransfer)
                    VSquared_s = np.abs(V_s)**2
                    VSquared_a = np.abs(V_a)**2
                    VSquaredArray_s[eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex] = VSquared_s
                    VSquaredArray_a[eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex] = VSquared_a
    for eigenstateAiIndex in range(numEStatesIncluded):
        eigenStateAiEnergy_s = Es_s[eigenstateAiIndex]
        eigenStateAiEnergy_a = Es_a[eigenstateAiIndex]
        for eigenstateAjIndex in range(numEStatesIncluded):
            eigenStateAjEnergy_s = Es_s[eigenstateAjIndex]
            eigenStateAjEnergy_a = Es_a[eigenstateAjIndex]
            for eigenstateBiIndex in range(numEStatesIncluded):
                eigenStateBiEnergy_s = Es_s[eigenstateBiIndex]
                eigenStateBiEnergy_a = Es_a[eigenstateBiIndex]
                for eigenstateBjIndex in range(numEStatesIncluded):
                    eigenStateBjEnergy_s = Es_s[eigenstateBjIndex]
                    eigenStateBjEnergy_a = Es_a[eigenstateBjIndex]
                    VSquared_s = VSquaredArray_s[eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex]
                    VSquared_a = VSquaredArray_a[eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex]
                    for TIndex in range(np.shape(T1T2Vals)[0]):
                        beta1, beta2 = beta1beta2Vals[TIndex]
                        Z1, Z2 = Z1Z2Vals[TIndex]
                        mu1 = muOfTVector1[TIndex]
                        mu2 = muOfTVector2[TIndex]
                        detailedBalance_s = computeDetailedBalance(eigenStateAiEnergy_s-2*mu1, eigenStateBiEnergy_s-2*mu2, beta1, beta2, Z1, Z2)
                        detailedBalance_a = computeDetailedBalance(eigenStateAiEnergy_a-2*mu1, eigenStateBiEnergy_a-2*mu2, beta1, beta2, Z1, Z2)
                        deltaEnergy_s = ((eigenStateAiEnergy_s-eigenStateAjEnergy_s)-(eigenStateBiEnergy_s-eigenStateBjEnergy_s))/2
                        deltaEnergy_a = ((eigenStateAiEnergy_a-eigenStateAjEnergy_a)-(eigenStateBiEnergy_a-eigenStateBjEnergy_a))/2
                        lorentzianFactor_s = computeLorentzianFactor(eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, Es_s, Es_a, eigenstateIndextoNArraySym, eigenstateIndextoNArrayAsym, 1, GammaofN, Gamma0)
                        lorentzianFactor_a = computeLorentzianFactor(eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, Es_s, Es_a, eigenstateIndextoNArraySym, eigenstateIndextoNArrayAsym, -1, GammaofN, Gamma0)
                        QDotFock2[TIndex] += VSquared_s * detailedBalance_s * deltaEnergy_s * lorentzianFactor_s + VSquared_a * detailedBalance_a * deltaEnergy_a * lorentzianFactor_a
    # fock 1 subspace
    QDotFock1 = np.zeros(np.shape(T1T2Vals)[0])
    for oneBodyStateAiIndex in range(numEStatesIncluded):
        oneBodyStateAi = oneBodyStatesMatrix[:, oneBodyStateAiIndex]
        oneBodyStateAiEnergy = EOneBody[oneBodyStateAiIndex]
        npAi=oneBodyStateAi[0]
        nmAi=oneBodyStateAi[1]
        for oneBodyStateAjIndex in range(numEStatesIncluded):
            oneBodyStateAj = oneBodyStatesMatrix[:, oneBodyStateAjIndex]
            oneBodyStateAjEnergy = EOneBody[oneBodyStateAjIndex]
            npAj=oneBodyStateAj[0]
            nmAj=oneBodyStateAj[1]
            for oneBodyStateBiIndex in range(numEStatesIncluded):
                oneBodyStateBi = oneBodyStatesMatrix[:, oneBodyStateBiIndex]
                oneBodyStateBiEnergy = EOneBody[oneBodyStateBiIndex]
                npBi=oneBodyStateBi[0]
                nmBi=oneBodyStateBi[1]
                for oneBodyStateBjIndex in range(numEStatesIncluded):
                    oneBodyStateBj = oneBodyStatesMatrix[:, oneBodyStateBjIndex]
                    oneBodyStateBjEnergy = EOneBody[oneBodyStateBjIndex]
                    npBj=oneBodyStateBj[0]
                    nmBj=oneBodyStateBj[1]
                    V=ed.Coul_hh_2body_individualQuantumNumbers_verticalSeparation(hbaromega,dTilde, npAi,nmAi,npBi,nmBi,npAj,nmAj,npBj,nmBj, mStar,epsForEnergyTransfer)
                    for TIndex in range(np.shape(T1T2Vals)[0]):
                        beta1, beta2 = beta1beta2Vals[TIndex]
                        Z1, Z2 = Z1Z2Vals[TIndex]
                        mu1 = muOfTVector1[TIndex]
                        mu2 = muOfTVector2[TIndex]
                        detailedBalance = computeDetailedBalance(oneBodyStateAiEnergy-mu1, oneBodyStateBiEnergy-mu2, beta1, beta2, Z1, Z2)
                        deltaEnergy = ((oneBodyStateAiEnergy-oneBodyStateAjEnergy)-(oneBodyStateBiEnergy-oneBodyStateBjEnergy))/2
                        detuning = abs(((oneBodyStateAiEnergy-oneBodyStateAjEnergy)+(oneBodyStateBiEnergy-oneBodyStateBjEnergy)))
                        GammaAi = GammaofN[npAi+nmAi] #doesnt work once bands start crossing, need to revise
                        GammaAj = GammaofN[npAj+nmAj] #doesnt work once bands start crossing
                        GammaBi = GammaofN[npBi+nmBi] #doesnt work once bands start crossing, need to revise
                        GammaBj = GammaofN[npBj+nmBj] #doesnt work once bands start crossing
                        Gamma = GammaAi+GammaAj+GammaBi+GammaBj+4*Gamma0
                        lorentzianFactor = (1/np.pi)*(Gamma/2)/((detuning)**2+(Gamma/2)**2)
                        QDotFock1[TIndex] += abs(V)**2 * detailedBalance * deltaEnergy * lorentzianFactor
    QDotFock1*=2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2 #(last factor of two is because two moire atoms per unit cell) # outputs in Watts
    QDotFock2*=2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2
    QDot = QDotFock1 + QDotFock2
    GFock1 = np.einsum('i,i-> i', abs(QDotFock1), abs((T1T2Vals[:,0]-T1T2Vals[:,1])**(-1)))
    GFock2 = np.einsum('i,i-> i', abs(QDotFock2), abs((T1T2Vals[:,0]-T1T2Vals[:,1])**(-1)))
    G = GFock1+GFock2
    return(QDot, G, GFock1, GFock2)

def computeLineCutInT(material, theta, nu, modStrengthFactor, N, numEStatesIncluded, TMin, TMax, numTVals, sMin, sMax, numsVals, dMin, dMax, numdVals, cutoff, muOfTPrecision, epsForED, epsForEnergyTransfer, epsilonEffD, Gamma0, numShellsForGammaofN, saveData):
    a, V1, V2, V3, phi, mStar = cmp.materialContinuumModelParameters(material)
    print()
    #print('a, V1, V2, V3, phi, mStar:', a, V1, V2, V3, phi, mStar)
    aM, gamma, hbaromega, l, coulombEnergy, ratio = computeParameters(theta, a, V1, V2, V3, mStar, epsForEnergyTransfer, modStrengthFactor)
    A = (aM**2) * math.sqrt(3)/2 # real space moire unit cell area
    #print('starting GammaofN')
    GammaofN = calculateGammaofN(theta, material, N, modStrengthFactor, numShellsForGammaofN)
    #print('finished GammaofN')
    sVals = np.linspace(sMin, sMax, numsVals)
    dVals = np.linspace(dMin, dMax, numdVals)
    dTildeVals = epsilonEffD * dVals/l
    sTildeVals = sVals/l
    print('theta:', theta)
    print('coulombEnergy:', coulombEnergy)
    print('hbaromega:', hbaromega)
    print('lambda:', coulombEnergy/hbaromega)
    #print('aM, gamma, hbaromega, l, coulombEnergy, ratio:', aM, gamma, hbaromega, l, coulombEnergy, ratio )
    print('GammaofN:', GammaofN)
    E0, basisStateList, E0_s, basisStateList_s, E0_a, basisStateList_a = ed.nonint_basis(N, hbaromega) #stateLists have each row a basis state with each column its quantum numbers (nR+,nRm,nr+,nr-)
    Es, Es_s, evecs_s, Es_a, evecs_a, parityList = ed.ED_hh(E0_s, E0_a, basisStateList_s, basisStateList_a, hbaromega, mStar,epsForED)
    #print('Es[0:10]:', Es[0:10]/hbaromega)
    eigenstateIndextoNArray = np.sum(basisStateList, axis=1).astype(int) # sum over the four quantum numbers of a basis states gives its energy in units of hbarOmega
    TLineCutVals = np.linspace(TMin,TMax,numTVals)
    T1T2Vals = np.vstack((1.01*TLineCutVals, 0.99*TLineCutVals)).transpose()
    beta1beta2Vals = (Kb*T1T2Vals)**(-1)
    Z1Z2Vals = np.zeros_like(T1T2Vals)
    if nu == 2:
        #only one calculation to do in this case, no interacting vs noninteracting
        EOneBody, stateListOneBody = ed.oneBody_basis(N, hbaromega)
        print('EOneBody[0:10]:', EOneBody[0:10]/hbaromega)
        print('starting muOfTVector1 \n')
        start = timeit.default_timer()
        muOfTVector1 = computeMuOfTVector(T1T2Vals[:,0], hbaromega, 1, nu, muOfTPrecision, EOneBody, Es_s, Es_a)
        muOfTVector2 = computeMuOfTVector(T1T2Vals[:,1], hbaromega, 1, nu, muOfTPrecision, EOneBody, Es_s, Es_a)
        stop = timeit.default_timer()
        print('time for muOfTVector1:', stop - start)
        for i in range(np.shape(Z1Z2Vals)[0]):
            Z1 = computePartitionFunction(nu, beta1beta2Vals[i,0], muOfTVector1[i], Es_s, Es_a, EOneBody)
            Z2 = computePartitionFunction(nu, beta1beta2Vals[i,1], muOfTVector2[i], Es_s, Es_a, EOneBody)
            Z1Z2Vals[i]=np.array([Z1,Z2])
        GLineCutInTofdandsNu2Fock1 = np.zeros((np.shape(dVals)[0], np.shape(sVals)[0], numTVals))
        GLineCutInTofdandsNu2Fock2 = np.zeros((np.shape(dVals)[0], np.shape(sVals)[0], numTVals))
        start = timeit.default_timer()
        for dIndex in range(np.shape(dVals)[0]):
            dTilde=dTildeVals[dIndex]
            for sIndex in range(np.shape(sVals)[0]):
                sTilde=sTildeVals[sIndex]
                QDot, GLineCutInTVals, GLineCutInTValsFock1, GLineCutInTValsFock2 = computeEnergyTransferTLineCutNu2(muOfTVector1, muOfTVector2, hbaromega, T1T2Vals,beta1beta2Vals, Z1Z2Vals, dTilde, mStar, epsForEnergyTransfer, numEStatesIncluded, cutoff, A, EOneBody, Es_s, Es_a, evecs_s, evecs_a, oneBodyStatesMatrix, basisStatesMatrix_s, basisStatesMatrix_a, GammaofN, Gamma0, eigenstateIndextoNArraySym, eigenstateIndextoNArrayAsym)
                GLineCutInTofdandsNu2Fock1[dIndex, sIndex] = GLineCutInTValsFock1
                GLineCutInTofdandsNu2Fock2[dIndex, sIndex] = GLineCutInTValsFock2
        stop = timeit.default_timer()
        GLineCutInTofdandsNu2 = GLineCutInTofdandsNu2Fock1+GLineCutInTofdandsNu2Fock2
        print('Time for GLineCutInT: ', stop - start)
        np.save(dataSaveDir+'GLineCutInTofdandsNu2%stheta=%sMod=%dEpsilonED=%ddMin=%ddMax=%dnumdVals=%dsMin=%dsMax=%dnumsVals=%dN=%dIncluded=%dTMin=%dTMax=%dnumTVals=%dGamma0=%d' % (date, theta, modStrengthFactor, epsForED, np.min(dVals), np.max(dVals), np.shape(dVals)[0],np.min(sVals), np.max(sVals), np.shape(sVals)[0], N, numEStatesIncluded, TMin, TMax, numTVals, Gamma0), GLineCutInTofdandsNu2)
        return(GLineCutInTofdandsNu2, GLineCutInTofdandsNu2Fock1, GLineCutInTofdandsNu2Fock2, muOfTVector1/hbaromega)
    if nu == 4:
        for i in range(np.shape(Z1Z2Vals)[0]):
            Z1 = computePartitionFunction(nu, beta1beta2Vals[i,0], 0, Es_s, Es_a)
            Z2 = computePartitionFunction(nu, beta1beta2Vals[i,1], 0, Es_s, Es_a)
            Z1Z2Vals[i]=np.array([Z1,Z2])
        # compute interacting and non-interacting separately
        GLineCutInTofdands = np.zeros((np.shape(dVals)[0], np.shape(sVals)[0], numTVals))
        ### Interacting with local two-body interaction
        start = timeit.default_timer()
        for dIndex in range(np.shape(dVals)[0]):
            dTilde=dTildeVals[dIndex]
            for sIndex in range(np.shape(sVals)[0]):
                sTilde=sTildeVals[sIndex]
                QDot, GLineCutInTVals = computeEnergyTransferTLineCut(hbaromega, T1T2Vals,beta1beta2Vals, Z1Z2Vals, dTilde, sTilde, mStar, epsForEnergyTransfer, numEStatesIncluded, cutoff, A, Es, evecs_s, evecs_a, parityList,  basisStateList_s, basisStateList_a, GammaofN, Gamma0, eigenstateIndextoNArray)
                print('GLineCutInTVals:', GLineCutInTVals*10**(-6))
                GLineCutInTofdands[dIndex, sIndex] = GLineCutInTVals
        stop = timeit.default_timer()
        print('Time for interacting: ', stop - start)
        #Noninteracting
        """
        GLineCutInTofdandsNoninteracting = np.zeros((np.shape(dVals)[0], np.shape(sVals)[0], numTVals))
        Z1Z2ValsNonint = np.zeros_like(T1T2Vals)
        for i in range(np.shape(Z1Z2ValsNonint)[0]):
            Z1 = computePartitionFunction(nu, beta1beta2Vals[i,0], 0, ESingleParticle_s, ESingleParticle_a)
            Z2 = computePartitionFunction(nu, beta1beta2Vals[i,1], 0, ESingleParticle_s, ESingleParticle_a)
            Z1Z2ValsNonint[i]=np.array([Z1,Z2])
        start = timeit.default_timer()
        for dIndex in range(np.shape(dVals)[0]):
            dTilde=dTildeVals[dIndex]
            for sIndex in range(np.shape(sVals)[0]):
                sTilde=dTildeVals[sIndex]
                QDot, GLineCutInTValsNoninteracting = computeEnergyTransferTLineCutSingleParticle(hbaromega, T1T2Vals,beta1beta2Vals, Z1Z2ValsNonint, dTilde, sTilde, mStar, epsForEnergyTransfer, numEStatesIncluded, A, ESingleParticle_s, ESingleParticle_a, basisStatesMatrix_s, basisStatesMatrix_a, GammaofN, Gamma0)
                GLineCutInTofdandsNoninteracting[dIndex, sIndex] = GLineCutInTValsNoninteracting
        stop = timeit.default_timer()
        print('Time for noninteracting: ', stop - start)
        """
        GLineCutInTofdandsNoninteracting = np.zeros_like(GLineCutInTofdands)
        GLineCutInTofdandsInteractingandSingleParticle = np.stack((GLineCutInTofdands, GLineCutInTofdandsNoninteracting))
        GLineCutInTofdandsInteractingandSingleParticleSaveName = str(dataSaveDir+'GLineCutInTofdandsInteractingandSingleParticle%stheta=%sMod=%dEpsilonED=%ddMin=%ddMax=%dnumdVals=%dsMin=%dsMax=%dnumsVals=%dN=%dIncluded=%dTMin=%dTMax=%dnumTVals=%dGamma0=%d' % (date, theta, modStrengthFactor, epsForED, np.min(dVals), np.max(dVals), np.shape(dVals)[0],np.min(sVals), np.max(sVals), np.shape(sVals)[0], N, numEStatesIncluded, TMin, TMax, numTVals, Gamma0))
        if saveData == True:
            np.save(GLineCutInTofdandsInteractingandSingleParticleSaveName, GLineCutInTofdandsInteractingandSingleParticle)
        return(GLineCutInTofdandsInteractingandSingleParticle)

#OLD VERSIONS AS OF 06/07/2022, WHICH DON'T ALLOW FOR TRIPLET-SINGLET COUPLING

"""
def computeEnergyTransferTLineCutSingleParticle(hbaromega, T1T2Vals,beta1beta2Vals, Z1Z2Vals, dTilde, sTilde, mStar, epsForEnergyTransfer, numEStatesIncluded, A, ESingleParticle_s, ESingleParticle_a, basisStatesMatrix_s, basisStatesMatrix_a, GammaofN, Gamma0):
    QDot_s = np.zeros(np.shape(T1T2Vals)[0])
    QDot_a = np.zeros(np.shape(T1T2Vals)[0])
    VSquaredArray_s = np.zeros((numEStatesIncluded, numEStatesIncluded, numEStatesIncluded, numEStatesIncluded))
    VSquaredArray_a = np.zeros((numEStatesIncluded, numEStatesIncluded, numEStatesIncluded, numEStatesIncluded))
    for basisStateAiIndex in range(numEStatesIncluded):
        basisStateAi_s = basisStatesMatrix_s[:, basisStateAiIndex]
        basisStateAiEnergy_s = ESingleParticle_s[basisStateAiIndex]
        nRpAi_s=basisStateAi_s[0]
        nRmAi_s=basisStateAi_s[1]
        nrpAi_s=basisStateAi_s[2]
        nrmAi_s=basisStateAi_s[3]
        basisStateAi_a = basisStatesMatrix_a[:, basisStateAiIndex]
        basisStateAiEnergy_a = ESingleParticle_a[basisStateAiIndex]
        nRpAi_a=basisStateAi_a[0]
        nRmAi_a=basisStateAi_a[1]
        nrpAi_a=basisStateAi_a[2]
        nrmAi_a=basisStateAi_a[3]
        for basisStateAjIndex in range(numEStatesIncluded):
            basisStateAj_s = basisStatesMatrix_s[:, basisStateAjIndex]
            basisStateAjEnergy_s = ESingleParticle_s[basisStateAjIndex]
            nRpAj_s=basisStateAj_s[0]
            nRmAj_s=basisStateAj_s[1]
            nrpAj_s=basisStateAj_s[2]
            nrmAj_s=basisStateAj_s[3]
            basisStateAj_a = basisStatesMatrix_a[:, basisStateAjIndex]
            basisStateAjEnergy_a = ESingleParticle_a[basisStateAjIndex]
            nRpAj_a=basisStateAj_a[0]
            nRmAj_a=basisStateAj_a[1]
            nrpAj_a=basisStateAj_a[2]
            nrmAj_a=basisStateAj_a[3]
            for basisStateBiIndex in range(numEStatesIncluded):
                basisStateBi_s = basisStatesMatrix_s[:, basisStateBiIndex]
                basisStateBiEnergy_s = ESingleParticle_s[basisStateBiIndex]
                nRpBi_s=basisStateBi_s[0]
                nRmBi_s=basisStateBi_s[1]
                nrpBi_s=basisStateBi_s[2]
                nrmBi_s=basisStateBi_s[3]
                basisStateBi_a = basisStatesMatrix_a[:, basisStateBiIndex]
                basisStateBiEnergy_a = ESingleParticle_a[basisStateBiIndex]
                nRpBi_a=basisStateBi_a[0]
                nRmBi_a=basisStateBi_a[1]
                nrpBi_a=basisStateBi_a[2]
                nrmBi_a=basisStateBi_a[3]
                for basisStateBjIndex in range(numEStatesIncluded):
                    basisStateBj_s = basisStatesMatrix_s[:, basisStateBjIndex]
                    basisStateBjEnergy_s = ESingleParticle_s[basisStateBjIndex]
                    nRpBj_s=basisStateBj_s[0]
                    nRmBj_s=basisStateBj_s[1]
                    nrpBj_s=basisStateBj_s[2]
                    nrmBj_s=basisStateBj_s[3]
                    basisStateBj_a = basisStatesMatrix_a[:, basisStateBjIndex]
                    basisStateBjEnergy_a = ESingleParticle_a[basisStateBjIndex]
                    nRpBj_a=basisStateBj_a[0]
                    nRmBj_a=basisStateBj_a[1]
                    nrpBj_a=basisStateBj_a[2]
                    nrmBj_a=basisStateBj_a[3]
                    V_s=ed.Coul_hh_4body_generalSeparation(dTilde,sTilde,hbaromega,nRpAi_s,nRmAi_s,nrpAi_s,nrmAi_s,nRpAj_s,nRmAj_s,nrpAj_s,nrmAj_s,nRpBi_s,nRmBi_s,nrpBi_s,nrmBi_s,nRpBj_s,nRmBj_s,nrpBj_s,nrmBj_s,mStar,epsForEnergyTransfer)
                    VSquaredArray_s[basisStateAiIndex, basisStateAjIndex, basisStateBiIndex, basisStateBjIndex]= abs(V_s)**2
                    V_a=ed.Coul_hh_4body_generalSeparation(dTilde,sTilde,hbaromega,nRpAi_a,nRmAi_a,nrpAi_a,nrmAi_a,nRpAj_a,nRmAj_a,nrpAj_a,nrmAj_a,nRpBi_a,nRmBi_a,nrpBi_a,nrmBi_a,nRpBj_a,nRmBj_a,nrpBj_a,nrmBj_a,mStar,epsForEnergyTransfer)
                    VSquaredArray_a[basisStateAiIndex, basisStateAjIndex, basisStateBiIndex, basisStateBjIndex]= abs(V_a)**2
    for basisStateAiIndex in range(numEStatesIncluded):
        basisStateAi_s = basisStatesMatrix_s[:, basisStateAiIndex]
        basisStateAiEnergy_s = ESingleParticle_s[basisStateAiIndex]
        NAi_s = int(1.01*basisStateAiEnergy_s/hbaromega-2)
        basisStateAi_a = basisStatesMatrix_a[:, basisStateAiIndex]
        basisStateAiEnergy_a = ESingleParticle_a[basisStateAiIndex]
        NAi_a = int(1.01*basisStateAiEnergy_a/hbaromega-2)
        for basisStateAjIndex in range(numEStatesIncluded):
            basisStateAj_s = basisStatesMatrix_s[:, basisStateAjIndex]
            basisStateAjEnergy_s = ESingleParticle_s[basisStateAjIndex]
            NAj_s = int(1.01*basisStateAjEnergy_s/hbaromega-2)
            basisStateAj_a = basisStatesMatrix_a[:, basisStateAjIndex]
            basisStateAjEnergy_a = ESingleParticle_a[basisStateAjIndex]
            NAj_a = int(1.01*basisStateAjEnergy_a/hbaromega-2)
            for basisStateBiIndex in range(numEStatesIncluded):
                basisStateBi_s = basisStatesMatrix_s[:, basisStateBiIndex]
                basisStateBiEnergy_s = ESingleParticle_s[basisStateBiIndex]
                NBi_s = int(1.01*basisStateBiEnergy_s/hbaromega-2)
                basisStateBi_a = basisStatesMatrix_a[:, basisStateBiIndex]
                basisStateBiEnergy_a = ESingleParticle_a[basisStateBiIndex]
                NBi_a = int(1.01*basisStateBiEnergy_a/hbaromega-2)
                for basisStateBjIndex in range(numEStatesIncluded):
                    basisStateBj_s = basisStatesMatrix_s[:, basisStateBjIndex]
                    basisStateBjEnergy_s = ESingleParticle_s[basisStateBjIndex]
                    NBj_s = int(1.01*basisStateBjEnergy_s/hbaromega-2)
                    basisStateBj_a = basisStatesMatrix_a[:, basisStateBjIndex]
                    basisStateBjEnergy_a = ESingleParticle_a[basisStateBjIndex]
                    NBj_a = int(1.01*basisStateBjEnergy_a/hbaromega-2)
                    VSquared_s = VSquaredArray_s[basisStateAiIndex,basisStateAjIndex,basisStateBiIndex,basisStateBjIndex]
                    VSquared_a = VSquaredArray_a[basisStateAiIndex,basisStateAjIndex,basisStateBiIndex,basisStateBjIndex]
                    for TIndex in range(np.shape(T1T2Vals)[0]):
                        beta1, beta2 = beta1beta2Vals[TIndex]
                        Z1, Z2 = Z1Z2Vals[TIndex]
                        detailedBalance_s = computeDetailedBalance(basisStateAiEnergy_s, basisStateBiEnergy_s, beta1, beta2, Z1, Z2)
                        detailedBalance_a = computeDetailedBalance(basisStateAiEnergy_a, basisStateBiEnergy_a, beta1, beta2, Z1, Z2)
                        deltaEnergy_s = ((basisStateAiEnergy_s-basisStateAjEnergy_s)-(basisStateBiEnergy_s-basisStateBjEnergy_s))/2
                        deltaEnergy_a = ((basisStateAiEnergy_a-basisStateAjEnergy_a)-(basisStateBiEnergy_a-basisStateBjEnergy_a))/2
                        detuning_s = abs((basisStateAiEnergy_s-basisStateAjEnergy_s)+(basisStateBiEnergy_s-basisStateBjEnergy_s))
                        detuning_a = abs((basisStateAiEnergy_a-basisStateAjEnergy_a)+(basisStateBiEnergy_a-basisStateBjEnergy_a))
                        GammaAi_s = GammaofN[NAi_s] #doesnt work once bands start crossing, need to revise
                        GammaAj_s = GammaofN[NAj_s]
                        GammaBi_s = GammaofN[NBi_s]
                        GammaBj_s = GammaofN[NBj_s] 
                        Gamma_s = GammaAi_s+GammaAj_s+GammaBi_s+GammaBj_s+4*Gamma0
                        GammaAi_a = GammaofN[NAi_a] #doesnt work once bands start crossing, need to revise
                        GammaAj_a = GammaofN[NAj_a] 
                        GammaBi_a = GammaofN[NBi_a]
                        GammaBj_a = GammaofN[NBj_a] 
                        Gamma_a = GammaAi_a+GammaAj_a+GammaBi_a+GammaBj_a+4*Gamma0
                        lorentzianFactor_s = (1/np.pi)*(Gamma_s/2)/((detuning_s)**2+(Gamma_s/2)**2)
                        lorentzianFactor_a = (1/np.pi)*(Gamma_a/2)/((detuning_a)**2+(Gamma_a/2)**2)
                        QDot_s[TIndex] += VSquared_s * detailedBalance_s * deltaEnergy_s * lorentzianFactor_s
                        QDot_a[TIndex] += VSquared_a * detailedBalance_a * deltaEnergy_a * lorentzianFactor_a
    QDot_s *= 2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2 #(last factor of two is because two moire atoms per unit cell) # outputs in Watts
    QDot_a *= 2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2
    QDot = QDot_s+QDot_a
    G_s = np.einsum('i,i-> i', abs(QDot_s), abs((T1T2Vals[:,0]-T1T2Vals[:,1])**(-1)))
    G_a = np.einsum('i,i-> i', abs(QDot_a), abs((T1T2Vals[:,0]-T1T2Vals[:,1])**(-1)))
    G = G_s + G_a
    return(QDot, G, G_s, G_a)

def computeEnergyTransferTLineCut(hbaromega, T1T2Vals,beta1beta2Vals, Z1Z2Vals, dTilde, sTilde, mStar, epsForEnergyTransfer, numEStatesIncluded, cutoff, A, Es_s, Es_a, evecs_s, evecs_a, basisStatesMatrix_s, basisStatesMatrix_a, GammaofN, Gamma0, eigenstateIndextoNArraySym, eigenstateIndextoNArrayAsym): #working as of 03/18/2022
    QDot_s = np.zeros(np.shape(T1T2Vals)[0])
    QDot_a = np.zeros(np.shape(T1T2Vals)[0])
    VSquaredArray_s = np.zeros((numEStatesIncluded, numEStatesIncluded, numEStatesIncluded, numEStatesIncluded))
    VSquaredArray_a = np.zeros((numEStatesIncluded, numEStatesIncluded, numEStatesIncluded, numEStatesIncluded))
    for eigenstateAiIndex in range(numEStatesIncluded):
        eigenStateAiEnergy_s = Es_s[eigenstateAiIndex]
        eigenStateAiEnergy_a = Es_a[eigenstateAiIndex]
        for eigenstateAjIndex in range(numEStatesIncluded):
            eigenStateAjEnergy_s = Es_s[eigenstateAjIndex]
            eigenStateAjEnergy_a = Es_a[eigenstateAjIndex]
            for eigenstateBiIndex in range(numEStatesIncluded):
                eigenStateBiEnergy_s = Es_s[eigenstateBiIndex]
                eigenStateBiEnergy_a = Es_a[eigenstateBiIndex]
                for eigenstateBjIndex in range(numEStatesIncluded):
                    eigenStateBjEnergy_s = Es_s[eigenstateBjIndex]
                    eigenStateBjEnergy_a = Es_a[eigenstateBjIndex]
                    V_s = ed.Coul_4body_estateestate(dTilde, sTilde, eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, cutoff, hbaromega, basisStatesMatrix_s, evecs_s, mStar, epsForEnergyTransfer)
                    V_a = ed.Coul_4body_estateestate(dTilde, sTilde, eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, cutoff, hbaromega, basisStatesMatrix_a, evecs_a, mStar, epsForEnergyTransfer)
                    VSquared_s = np.abs(V_s)**2
                    VSquared_a = np.abs(V_a)**2
                    VSquaredArray_s[eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex] = VSquared_s
                    VSquaredArray_a[eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex] = VSquared_a
    for eigenstateAiIndex in range(numEStatesIncluded):
        eigenStateAiEnergy_s = Es_s[eigenstateAiIndex]
        eigenStateAiEnergy_a = Es_a[eigenstateAiIndex]
        for eigenstateAjIndex in range(numEStatesIncluded):
            eigenStateAjEnergy_s = Es_s[eigenstateAjIndex]
            eigenStateAjEnergy_a = Es_a[eigenstateAjIndex]
            for eigenstateBiIndex in range(numEStatesIncluded):
                eigenStateBiEnergy_s = Es_s[eigenstateBiIndex]
                eigenStateBiEnergy_a = Es_a[eigenstateBiIndex]
                for eigenstateBjIndex in range(numEStatesIncluded):
                    eigenStateBjEnergy_s = Es_s[eigenstateBjIndex]
                    eigenStateBjEnergy_a = Es_a[eigenstateBjIndex]
                    VSquared_s = VSquaredArray_s[eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex]
                    VSquared_a = VSquaredArray_a[eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex]
                    for TIndex in range(np.shape(T1T2Vals)[0]):
                        beta1, beta2 = beta1beta2Vals[TIndex]
                        Z1, Z2 = Z1Z2Vals[TIndex]
                        detailedBalance_s = computeDetailedBalance(eigenStateAiEnergy_s, eigenStateBiEnergy_s, beta1, beta2, Z1, Z2)
                        detailedBalance_a = computeDetailedBalance(eigenStateAiEnergy_a, eigenStateBiEnergy_a, beta1, beta2, Z1, Z2)
                        deltaEnergy_s = ((eigenStateAiEnergy_s-eigenStateAjEnergy_s)-(eigenStateBiEnergy_s-eigenStateBjEnergy_s))/2
                        deltaEnergy_a = ((eigenStateAiEnergy_a-eigenStateAjEnergy_a)-(eigenStateBiEnergy_a-eigenStateBjEnergy_a))/2
                        lorentzianFactor_s = computeLorentzianFactor(eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, Es_s, Es_a, eigenstateIndextoNArraySym, eigenstateIndextoNArrayAsym, 1, GammaofN, Gamma0)
                        lorentzianFactor_a = computeLorentzianFactor(eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, Es_s, Es_a, eigenstateIndextoNArraySym, eigenstateIndextoNArrayAsym, -1, GammaofN, Gamma0)
                        QDot_s[TIndex] += VSquared_s * detailedBalance_s * deltaEnergy_s * lorentzianFactor_s
                        QDot_a[TIndex] += VSquared_a * detailedBalance_a * deltaEnergy_a * lorentzianFactor_a
    QDot_s *= 2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2 #(last factor of two is because two moire atoms per unit cell) # outputs in Watts
    QDot_a *= 2*np.pi/hbar/(A * SquareMetersPerSqareAngstrom) * JoulesPermeV * 2
    QDot = QDot_s+QDot_a
    G_s = np.einsum('i,i-> i', abs(QDot_s), abs((T1T2Vals[:,0]-T1T2Vals[:,1])**(-1)))
    G_a = np.einsum('i,i-> i', abs(QDot_a), abs((T1T2Vals[:,0]-T1T2Vals[:,1])**(-1)))
    G = G_s+G_a
    return(QDot, G, G_s, G_a)
"""

"""  
if computeGLineCutindFromScratch == True:
    #dLine cut
    #interacting
    TMax = TLineCutVals[np.argmax(np.abs(GLineCutInTofdandsInteractingandSingleParticle[0,0,0]))]
    print('index:', np.argmax(np.abs(GLineCutInTofdandsInteractingandSingleParticle[0,0,0])))
    print('TMax:', TMax)
    T1T2ValsFordLineCut = np.array([[1.01*TMax,0.99*TMax]])
    beta1beta2ValsFordLineCut = 1/(Kb*T1T2ValsFordLineCut)
    beta1 = beta1beta2ValsFordLineCut[0,0]
    beta2 = beta1beta2ValsFordLineCut[0,1]
    Z1 = computePartitionFunction(beta1)
    Z2 = computePartitionFunction(beta2)
    Z1Z2ValsValsFordLineCut = np.zeros((1,2))
    Z1Z2ValsValsFordLineCut[0,0]+=Z1
    Z1Z2ValsValsFordLineCut[0,1]+=Z2
    #noninteracting
    TMaxNoninteracting = TLineCutVals[np.argmax(np.abs(GLineCutInTofdandsInteractingandSingleParticle[1,0,0]))]
    print('TMaxNonInt:', TMaxNoninteracting)
    T1T2ValsForsLineCutNoninteracting  = np.array([[1.01*TMaxNoninteracting ,0.99*TMaxNoninteracting]])
    beta1beta2ValsForsLineCutNoninteracting  = 1/(Kb*T1T2ValsForsLineCutNoninteracting )
    beta1NonInt = beta1beta2ValsForsLineCutNoninteracting[0,0]
    beta2NonInt = beta1beta2ValsForsLineCutNoninteracting[0,1]
    Z1NonInt=np.exp(4*hbaromega*beta1NonInt)/(np.exp(hbaromega*beta1NonInt)-1)**4 * np.exp(-2*hbaromega*beta1NonInt)
    Z2NonInt=np.exp(4*hbaromega*beta2NonInt)/(np.exp(hbaromega*beta2NonInt)-1)**4 * np.exp(-2*hbaromega*beta2NonInt)
    Z1Z2ValsValsForsLineCutNoninteracting = np.zeros((1,2))
    Z1Z2ValsValsForsLineCutNoninteracting[0,0]+=Z1NonInt
    Z1Z2ValsValsForsLineCutNoninteracting[0,1]+=Z2NonInt
    dLineCutMin = np.log10(0.1)
    dLineCutMax = np.log10(40)
    numdLineCutVals = 20
    s = 0
    sTilde = s/l
    dLineCutVals = np.logspace(dLineCutMin,dLineCutMax,numdLineCutVals)
    GValsofdInteracting = np.zeros_like(dLineCutVals)
    for dIndex in range(numdLineCutVals):
        print('dIndex:', dIndex)
        d = dLineCutVals[dIndex]
        dTilde = epsilonEffD*d/l
        QDot, G = computeEnergyTransferTLineCut(T1T2ValsFordLineCut,beta1beta2ValsFordLineCut, Z1Z2ValsValsFordLineCut, dTilde, sTilde)
        GValsofdInteracting[dIndex] = G
    GValsNoninteractingofd = np.zeros_like(dLineCutVals)
    for dIndex in range(numdLineCutVals):
        print('dIndex:', dIndex)
        d = dLineCutVals[dIndex]
        dTilde = epsilonEffD*d/l
        QDot, G = computeEnergyTransferTLineCutSingleParticle(T1T2ValsForsLineCutNoninteracting,beta1beta2ValsForsLineCutNoninteracting, Z1Z2ValsValsForsLineCutNoninteracting, dTilde, sTilde)
        GValsNoninteractingofd[dIndex] = G
    GValsNoninteractingandSingleParticleofd = np.stack((GValsofdInteracting,GValsNoninteractingofd))
    GLineCutIndValsInteractingandSingleParticleSaveName = str('GLineCutIndInteractingandSingleParticleLogLog%stheta=%sMod=%dEpsilonED=%dTMaxInt=%dTMaxNonInt=%ddMin=%ddMax=%dnumdVal=%dN=%dIncluded=%d' % (date, theta, modStrengthFactor, epsForED, TMax, TMaxNoninteracting, dLineCutMin, dLineCutMax, numdLineCutVals, N, numEStatesIncluded))
    np.save(GLineCutIndValsInteractingandSingleParticleSaveName, GValsNoninteractingandSingleParticleofd)

if computeGLineCutinsFromScratch == True:
    sMin = 0
    sMax = aM/3
    numsVals = 5
    dForSLineCut=20
    #interacting
    TMax = TLineCutVals[np.argmax(GLineCutInTofdandsInteractingandSingleParticle[0,0,0])]
    T1T2ValsForsLineCut = np.array([[1.01*TMax,0.99*TMax]])
    beta1beta2ValsForsLineCut = 1/(Kb*T1T2ValsForsLineCut)
    beta1 = beta1beta2ValsForsLineCut[0,0]
    beta2 = beta1beta2ValsForsLineCut[0,1]
    Z1 = computePartitionFunction(beta1)
    Z2 = computePartitionFunction(beta2)
    Z1Z2ValsValsForsLineCut = np.zeros((1,2))
    Z1Z2ValsValsForsLineCut[0,0]+=Z1
    Z1Z2ValsValsForsLineCut[0,1]+=Z2
    #noninteracting
    #noninteracting
    TMaxNoninteracting = TLineCutVals[np.argmax(GLineCutInTofdandsInteractingandSingleParticle[1,0,0])]
    T1T2ValsForsLineCutNoninteracting  = np.array([[1.01*TMaxNoninteracting ,0.99*TMaxNoninteracting]])
    beta1beta2ValsForsLineCutNoninteracting  = 1/(Kb*T1T2ValsForsLineCutNoninteracting )
    beta1NonInt = beta1beta2ValsForsLineCutNoninteracting[0,0]
    beta2NonInt = beta1beta2ValsForsLineCutNoninteracting[0,1]
    Z1NonInt=np.exp(4*hbaromega*beta1NonInt)/(np.exp(hbaromega*beta1NonInt)-1)**4 * np.exp(-2*hbaromega*beta1NonInt)
    Z2NonInt=np.exp(4*hbaromega*beta2NonInt)/(np.exp(hbaromega*beta2NonInt)-1)**4 * np.exp(-2*hbaromega*beta2NonInt)
    Z1Z2ValsValsForsLineCutNoninteracting = np.zeros((1,2))
    Z1Z2ValsValsForsLineCutNoninteracting[0,0]+=Z1NonInt
    Z1Z2ValsValsForsLineCutNoninteracting[0,1]+=Z2NonInt
    sVals= np.linspace(sMin, sMax, numsVals)
    GValsofs = np.zeros_like(sVals)
    dtildeForSLineCut=epsilonEffD*dForSLineCut/l
    GValsofsInt = np.zeros_like(sVals)
    GValsofsSingleParticle = np.zeros_like(sVals)
    print('starting s line cut...')
    for sIndex in range(numsVals):
        print('sIndex:', sIndex)
        s = sVals[sIndex]
        sTilde = s/l
        QDotInt, GInt = computeEnergyTransferTLineCut(T1T2ValsForsLineCut,beta1beta2ValsForsLineCut, Z1Z2ValsValsForsLineCut,dtildeForSLineCut, sTilde)
        GValsofsInt[sIndex] = GInt
        QDotNonSingleParticle, GSingleParticle = computeEnergyTransferTLineCutSingleParticle(T1T2ValsForsLineCut,beta1beta2ValsForsLineCutNoninteracting, Z1Z2ValsValsForsLineCutNoninteracting, dtildeForSLineCut, s)
        GValsofsSingleParticle[sIndex] = GSingleParticle
    GValsNoninteractingandSingleParticleofs = np.stack((GValsofsInt,GValsofsSingleParticle))
    GValsNoninteractingandSingleParticleofsName = str('GValsNoninteractingandSingleParticleofs%stheta=%sEpsilonED=%dTMaxInt=%dTMaxNonInt=%dsMin=%dsMax=%dnumsVal=%dd=%dN=%dIncluded=%d' % (date, theta, epsForED, TMax, TMaxNoninteracting, sMin, sMax, numsVals, dForSLineCut, N, numEStatesIncluded))
    np.save(GValsNoninteractingandSingleParticleofsName, GValsNoninteractingandSingleParticleofs)
"""



"""
### Overkill partition function, including extra states in the sum

N_partitionFunction = 10

basis_s_partitionFunction = ed.nonint_basis_symmetric(N_partitionFunction,hbaromega)

basis_a_partitionFunction = ed.nonint_basis_antisymmetric(N_partitionFunction,hbaromega)

states_s_partitionFunction = np.concatenate(basis_s_partitionFunction[1:5]).flatten()

states_a_partitionFunction = np.concatenate(basis_a_partitionFunction[1:5]).flatten()

basisStatesMatrix_s_partitionFunction = np.reshape(states_s_partitionFunction, (4,np.shape(basis_s_partitionFunction[0])[0])) #basis states matrix has each column a direct product state with each row its quantum numbers (n1+,n1-,n2+,n2-)

basisStatesMatrix_a_partitionFunction = np.reshape(states_a_partitionFunction, (4,np.shape(basis_a_partitionFunction[0])[0]))

Es_s_partitionFunction,evecs_s_partitionFunction,Es_a_partitionFunction,evecs_a_partitionFunction = ed.ED_hh(basis_s_partitionFunction, basis_a_partitionFunction, hbaromega, epsForED)

@njit
def computePartitionFunction(beta):
    Z=0
    for energyVal_s_partitionFunction in Es_s_partitionFunction:
        Z+=np.exp(-1*(energyVal_s_partitionFunction)*beta)
    for energyVal_a_partitionFunction in Es_a_partitionFunction:
        Z+=np.exp(-1*(energyVal_a_partitionFunction)*beta)
    return(Z)
"""

"""
#analytic noninteracting partition funciton
@njit
def computePartitionFunction(beta):
    Z=np.exp(4*hbaromega*beta)/(np.exp(hbaromega*beta)-1)**4 * np.exp(-2*hbaromega*beta)
    return(Z)
"""