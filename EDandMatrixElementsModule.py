from numpy import *
import scipy.linalg as linalg
import numba
import scipy
from scipy.special import gamma, hyperu, assoc_laguerre
import scipy.integrate as integrate
import math
from math import isclose
import mpmath as mp

"""
Note: numba is not compatible mpmath, so I cannot use numba on the matrix element calculations that use the hyperu function from mpmath.
"""

hbar = 6.582 * 10**(-13) # meV * s
electronMass = 5.68563 * 10**(-29) # meV *(second/Å)
eSquaredOvere0 =  14400 #meV * angstrom #CGS

def oneBody_basis(N, omgh):
    numBasisStates = int((N+1)*(N+2)/2)
    stateList = zeros((numBasisStates,2))
    npList = zeros(numBasisStates)
    nmList = zeros(numBasisStates)
    index = 0
    for np in range(N+1):
        for nm in range(N-np+1):
            stateList[index] = np.array([np,nm])
            index += 1
    E0 = (sum(stateList, axis=1) + 1)*omgh
    ind_sort = argsort(E0)
    E0_sorted = E0[ind_sort]
    stateList_sorted=stateList[ind_sort]
    return(E0,stateList_sorted.astype(int))

def nonint_basis(N, omgh):
    numBasisStates = int((1/24)*((N+1)*(N+2)*(N+3)*(N+4))) # the number of 2 particle 2DIHO states at or below the the N^th noninteracting energy level
    #First construct a basis with size (nhp*nhm)**2, then pick out Ncut lowest states (Ncut is the number of two-particle basis states we retain)
    stateList = zeros((numBasisStates, 4)) # rows are states, columns are (nRp, nRm, nrp, nrm)
    index = 0
    for nRp in range(N+1):
        for nRm in range(N-nRp+1):
            for nrp in range(N-nRp-nRm+1):
                for nrm in range(N-nRp-nRm-nrp+1):
                    stateList[index] = array((nRp, nRm, nrp, nrm))
                    index += 1
    E0 = (sum(stateList, axis=1)+2)*omgh
    ind_sort = argsort(E0)
    E0_sorted = E0[ind_sort]
    stateList_sorted = stateList[ind_sort]
    nrpList = stateList_sorted[:,2]
    nrmList = stateList_sorted[:,3]
    ind_s = abs(nrpList-nrmList)%2 == 0 #_s is symmetric, _a is antisymmetric
    E0_s = E0_sorted[ind_s]
    stateList_s = stateList_sorted[ind_s]
    ind_a = abs(nrpList-nrmList)%2 == 1 #_s is symmetric, _a is antisymmetric
    E0_a = E0_sorted[ind_a]
    stateList_a = stateList_sorted[ind_a]
    return(E0_sorted, stateList_sorted.astype(int),E0_s, stateList_s.astype(int), E0_a, stateList_a.astype(int))

@numba.jit()
def factorial(n): #a factorial function with floating number output, to avoid numerical issue
    x = 1.0
    for m in range(1,n+1):
        x *= m
    return x

@numba.jit()
def Coul_hh_2body_noSeparation(omgh,nRpi,nRmi,nrpi,nrmi,nRpj,nRmj,nrpj,nrmj, mStar, dielectricConstant): #this is for two holes on the same quantum dot
    #h-h Coulomb matrix element, two holes distinguishable
    #<nRpi,nRmi;nrpi,nrmi|V_hh|nRpj,nRmj;nrpj,nrmj>
    nsum = nrpj+nrmj+nrmi+nrpi
    dl = (nrpj-nrmj)-(nrpi-nrmi)
    deltaNr = (nrpj+nrmj)-(nrpi+nrmi)
    if dl!=0 or nRpi != nRpj or nRmi != nRmj:
        return 0 #angular momentum conservation
    Srp = 0
    arp = 1/factorial(nrpi)/factorial(nrpj)
    for krp in range(min(nrpi,nrpj)+1):
        Srm = 0
        arm = 1/factorial(nrmi)/factorial(nrmj)
        for krm in range(min(nrmi,nrmj)+1):
            p = nsum-2*(krp+krm)
            Srm += arm*math.gamma((p + 1)/2)/2
            arm *= -1*(nrmi-krm)*(nrmj-krm)/(krm+1)
        Srp += arp*Srm
        arp *= -1*(nrpi-krp)*(nrpj-krp)/(krp+1)
    L = sqrt(hbar**2/(omgh*mStar*electronMass))
    E0 = eSquaredOvere0/(dielectricConstant*L)
    # note that ((-1)**(deltaNr/2)) = (1j)**(deltaNr) since deltaNr is necessarily even. I just code it here as the LHS of this equation to ensure that it is real for the diagonalization code.
    Vij = Srp*sqrt(2)*E0*((-1)**(deltaNr/2))*((-1)**(nrpi+nrmi))*sqrt(factorial(nrpi)*factorial(nrpj)*factorial(nrmi)*factorial(nrmj))
    return Vij

def Coul_hh_2body_individualQuantumNumbers_verticalSeparation(omgh,dTilde,n1pi,n1mi,n2pi,n2mi,n1pj,n1mj,n2pj,n2mj, mStar,dielectricConstant): #this is for different quantum dots each with one hole
    if dTilde == 0:
        return(Coul_hh_2body_individualQuantumNumbers_noSeparation(omgh, n1pi,n1mi,n2pi,n2mi,n1pj,n1mj,n2pj,n2mj, mStar,dielectricConstant))
    nsum = n1pi+n1mi+n2pi+n2mi+n1pj+n1mj+n2pj+n2mj
    dl = (n1pi+n2pi-n1mi-n2mi)-(n1pj+n2pj-n1mj-n2mj)
    deltaN2 = (n2pi+n2mi)-(n2pj+n2mj)
    deltaN1 = (n1pi+n1mi)-(n1pj+n1mj)
    deltaNTot = deltaN1 + deltaN2
    if dl!=0:
        return 0 #angular momentum conservation
    S1p = 0 #sum over k1p
    a1p = 1/factorial(n1pi)/factorial(n1pj) #prefactor of each term in S1p
    for k1p in range(min(n1pi,n1pj)+1):
        S1m = 0 #sum over k1m
        a1m = 1/factorial(n1mi)/factorial(n1mj)
        for k1m in range(min(n1mi,n1mj)+1):
            S2p = 0
            a2p = 1/factorial(n2pi)/factorial(n2pj)
            for k2p in range(min(n2pi,n2pj)+1):
                S2m = 0
                a2m = 1/factorial(n2mi)/factorial(n2mj)
                for k2m in range(min(n2mi,n2mj)+1):
                    p = nsum-2*(k1p+k1m+k2p+k2m)
                    I = sqrt(2)**(-3*(p+1)) * math.gamma(p+1)*mp.hyperu((p+1)/2, (1/2), (1/2)*(dTilde)**2)
                    S2m += a2m*I
                    a2m *= -2*(n2mi-k2m)*(n2mj-k2m)/(k2m+1)
                S2p += a2p*S2m
                a2p *= -2*(n2pi-k2p)*(n2pj-k2p)/(k2p+1)
            S1m += a1m*S2p
            a1m *= -2*(n1mi-k1m)*(n1mj-k1m)/(k1m+1)
        S1p += a1p*S1m
        a1p *= -2*(n1pi-k1p)*(n1pj-k1p)/(k1p+1)
    L = sqrt(hbar**2/(omgh*mStar*electronMass))
    E0 = eSquaredOvere0/(dielectricConstant*L)
    # note that ((-1)**(deltaNr/2)) = (1j)**(deltaNr) since deltaNr is necessarily even. I just code it here as the LHS of this equation to ensure that it is real for the diagonalization code.
    Vij = S1p*2*E0*exp((1j*pi/2)*(deltaNTot))*(-1)**(abs(deltaN2))*((-1)**abs(n1pj+n1mj+n2pj+n2mj))*sqrt(factorial(n1pi)*factorial(n1pj)*factorial(n1mi)*factorial(n1mj)*factorial(n2pi)*factorial(n2pj)*factorial(n2mi)*factorial(n2mj))
    return Vij

@numba.jit()
def Coul_hh_2body_individualQuantumNumbers_noSeparation(omgh, n1pi,n1mi,n2pi,n2mi,n1pj,n1mj,n2pj,n2mj, mStar,dielectricConstant): #this is for different quantum dots each with one hole
    nsum = n1pi+n1mi+n2pi+n2mi+n1pj+n1mj+n2pj+n2mj
    dl = (n1pi+n2pi-n1mi-n2mi)-(n1pj+n2pj-n1mj-n2mj)
    deltaN2 = (n2pi+n2mi)-(n2pj+n2mj)
    deltaN1 = (n1pi+n1mi)-(n1pj+n1mj)
    deltaNTot = deltaN1 + deltaN2
    if dl!=0:
        return 0 #angular momentum conservation
    S1p = 0 #sum over k1p
    a1p = 1/factorial(n1pi)/factorial(n1pj) #prefactor of each term in S1p
    for k1p in range(min(n1pi,n1pj)+1):
        S1m = 0 #sum over k1m
        a1m = 1/factorial(n1mi)/factorial(n1mj)
        for k1m in range(min(n1mi,n1mj)+1):
            S2p = 0
            a2p = 1/factorial(n2pi)/factorial(n2pj)
            for k2p in range(min(n2pi,n2pj)+1):
                S2m = 0
                a2m = 1/factorial(n2mi)/factorial(n2mj)
                for k2m in range(min(n2mi,n2mj)+1):
                    p = nsum-2*(k1p+k1m+k2p+k2m)
                    I = 2**(-(p+3)/2)*math.gamma((p+1)/2)
                    S2m += a2m*I
                    a2m *= -2*(n2mi-k2m)*(n2mj-k2m)/(k2m+1)
                S2p += a2p*S2m
                a2p *= -2*(n2pi-k2p)*(n2pj-k2p)/(k2p+1)
            S1m += a1m*S2p
            a1m *= -2*(n1mi-k1m)*(n1mj-k1m)/(k1m+1)
        S1p += a1p*S1m
        a1p *= -2*(n1pi-k1p)*(n1pj-k1p)/(k1p+1)
    L = sqrt(hbar**2/(omgh*mStar*electronMass))
    E0 = eSquaredOvere0/(dielectricConstant*L)
    # note that ((-1)**(deltaNr/2)) = (1j)**(deltaNr) since deltaNr is necessarily even. I just code it here as the LHS of this equation to ensure that it is real for the diagonalization code.
    Vij = S1p*2*E0*exp((1j*pi/2)*(deltaNTot))*((-1)**(deltaN2))*((-1)**(n1pj+n1mj+n2pj+n2mj))*sqrt(factorial(n1pi)*factorial(n1pj)*factorial(n1mi)*factorial(n1mj)*factorial(n2pi)*factorial(n2pj)*factorial(n2mi)*factorial(n2mj))
    return Vij

def Coul_hh_4body_generalSeparation(dTilde,sTilde,omgh,nRpAi,nRmAi,nrpAi,nrmAi,nRpAj,nRmAj,nrpAj,nrmAj,nRpBi,nRmBi,nrpBi,nrmBi,nRpBj,nRmBj,nrpBj,nrmBj,mStar,dielectricConstant):
    if sTilde == 0:
        return Coul_hh_4body_verticalSeparation(dTilde,omgh,nRpAi,nRmAi,nrpAi,nrmAi,nRpAj,nRmAj,nrpAj,nrmAj,nRpBi,nRmBi,nrpBi,nrmBi,nRpBj,nRmBj,nrpBj,nrmBj,mStar,dielectricConstant)
    nsum = nRpAi+nRmAi+nrpAi+nrmAi+nRpAj+nRmAj+nrpAj+nrmAj+nRpBi+nRmBi+nrpBi+nrmBi+nRpBj+nRmBj+nrpBj+nrmBj
    dl = (nRpAj-nRpAi)-(nRmAj-nRmAi)+(nrpAj-nrpAi)-(nrmAj-nrmAi)+(nRpBj-nRpBi)-(nRmBj-nRmBi)+(nrpBj-nrpBi)-(nrmBj-nrmBi)
    dlrA = (nrpAj-nrpAi)-(nrmAj-nrmAi)
    dlrB = (nrpBj-nrpBi)-(nrmBj-nrmBi)
    if dlrA%2 != 0 or dlrB%2 != 0:
        return 0
    SRpA = 0
    aRpA = 1/factorial(nRpAi)/factorial(nRpAj)
    for kRpA in range(min(nRpAi,nRpAj)+1):
        SRmA = 0
        aRmA = 1/factorial(nRmAi)/factorial(nRmAj)
        for kRmA in range(min(nRmAi,nRmAj)+1):
            SrpA = 0
            arpA = 1/factorial(nrpAi)/factorial(nrpAj)
            for krpA in range(min(nrpAi,nrpAj)+1):
                SrmA = 0
                armA = 1/factorial(nrmAi)/factorial(nrmAj)
                for krmA in range(min(nrmAi,nrmAj)+1):
                    SRpB = 0
                    aRpB = 1/factorial(nRpBi)/factorial(nRpBj)
                    for kRpB in range(min(nRpBi,nRpBj)+1):
                        SRmB = 0
                        aRmB = 1/factorial(nRmBi)/factorial(nRmBj)
                        for kRmB in range(min(nRmBi, nRmBj)+1):
                            SrpB = 0
                            arpB = 1/factorial(nrpBi)/factorial(nrpBj)
                            for krpB in range(min(nrpBi,nrpBj)+1):
                                SrmB = 0
                                armB = 1/factorial(nrmBi)/factorial(nrmBj)
                                for krmB in range(min(nrmBi,nrmBj)+1):
                                    pPrime = nsum-2*(kRpA+kRmA+krpA+krmA+kRpB+kRmB+krpB+krmB)
                                    if pPrime+1 < 0:
                                        print('hit!')
                                        print('nSum:', nsum)
                                        print('kSum:', kRpA+kRmA+krpA+krmA+kRpB+kRmB+krpB+krmB)
                                    IPrime = (1/4)**(pPrime+1)*math.gamma(pPrime+1)*(1/(2*pi))*real(mp.quad(lambda x: mp.hyperu((pPrime+1)/2, (1/2), (1/2)*((dTilde-1j*sTilde*mp.cos(x)))**2) * (1j*mp.exp(1j*x))**(dl), [0, 2*math.pi]))
                                    SrmB += armB*IPrime
                                    armB *= -1*(nrmBi-krmB)*(nrmBj-krmB)/(krmB+1)
                                SrpB += arpB*SrmB
                                arpB *= -1*(nrpBi-krpB)*(nrpBj-krpB)/(krpB+1)
                            SRmB += aRmB*SrpB
                            aRmB *= -1*(nRmBi-kRmB)*(nRmBj-kRmB)/(kRmB+1)
                        SRpB += aRpB*SRmB
                        aRpB *= -1*(nRpBi-kRpB)*(nRpBj-kRpB)/(kRpB+1)
                    SrmA += armA*SRpB
                    armA *= -1*(nrmAi-krmA)*(nrmAj-krmA)/(krmA+1)
                SrpA += arpA*SrmA
                arpA *= -1*(nrpAi-krpA)*(nrpAj-krpA)/(krpA+1)
            SRmA += aRmA*SrpA
            aRmA *= -1*(nRmAi-kRmA)*(nRmAj-kRmA)/(kRmA+1)
        SRpA += aRpA*SRmA
        aRpA *= -1*(nRpAi-kRpA)*(nRpAj-kRpA)/(kRpA+1)
    L = sqrt(hbar**2/(omgh*mStar*electronMass))
    E0 = eSquaredOvere0/(dielectricConstant*L)
    deltaNRB = nRmBi + nRpBi - nRmBj - nRpBj
    deltaNRA = nRmAi + nRpAi - nRmAj - nRpAj
    deltaNrB = nrmBi + nrpBi - nrmBj - nrpBj
    deltaNrA = nrmAi + nrpAi - nrmAj - nrpAj
    V = SRpA*2**(7/2)*E0*(-1)**(abs(deltaNRB+nRpAj+nRmAj+nrpAj+nrmAj+nRpBj+nRmBj+nrmBj+nrpBj))*1j**(abs(deltaNRB+deltaNRA+deltaNrB+deltaNrA))*sqrt(factorial(nrpAi)*factorial(nrpAj)*factorial(nrmAi)*factorial(nrmAj)*factorial(nrpBi)*factorial(nrpBj)*factorial(nrmBi)*factorial(nrmBj))
    return(V)

def Coul_hh_4body_verticalSeparation(dTilde,omgh,nRpAi,nRmAi,nrpAi,nrmAi,nRpAj,nRmAj,nrpAj,nrmAj,nRpBi,nRmBi,nrpBi,nrmBi,nRpBj,nRmBj,nrpBj,nrmBj,mStar,dielectricConstant):
    if dTilde == 0:
        return Coul_hh_4body_noSeparation(omgh,nRpAi,nRmAi,nrpAi,nrmAi,nRpAj,nRmAj,nrpAj,nrmAj,nRpBi,nRmBi,nrpBi,nrmBi,nRpBj,nRmBj,nrpBj,nrmBj,mStar,dielectricConstant)
    nsum = nRpAi+nRmAi+nrpAi+nrmAi+nRpAj+nRmAj+nrpAj+nrmAj+nRpBi+nRmBi+nrpBi+nrmBi+nRpBj+nRmBj+nrpBj+nrmBj
    dl = (nRpAj-nRpAi)-(nRmAj-nRmAi)+(nrpAj-nrpAi)-(nrmAj-nrmAi)+(nRpBj-nRpBi)-(nRmBj-nRmBi)+(nrpBj-nrpBi)-(nrmBj-nrmBi)
    if dl !=0:
        return 0
    dlrA = (nrpAj-nrpAi)-(nrmAj-nrmAi)
    dlrB = (nrpBj-nrpBi)-(nrmBj-nrmBi)
    if dlrA%2 != 0 or dlrB%2 != 0:
        return 0
    SRpA = 0
    aRpA = 1/factorial(nRpAi)/factorial(nRpAj)
    for kRpA in range(min(nRpAi,nRpAj)+1):
        SRmA = 0
        aRmA = 1/factorial(nRmAi)/factorial(nRmAj)
        for kRmA in range(min(nRmAi,nRmAj)+1):
            SrpA = 0
            arpA = 1/factorial(nrpAi)/factorial(nrpAj)
            for krpA in range(min(nrpAi,nrpAj)+1):
                SrmA = 0
                armA = 1/factorial(nrmAi)/factorial(nrmAj)
                for krmA in range(min(nrmAi,nrmAj)+1):
                    SRpB = 0
                    aRpB = 1/factorial(nRpBi)/factorial(nRpBj)
                    for kRpB in range(min(nRpBi,nRpBj)+1):
                        SRmB = 0
                        aRmB = 1/factorial(nRmBi)/factorial(nRmBj)
                        for kRmB in range(min(nRmBi, nRmBj)+1):
                            SrpB = 0
                            arpB = 1/factorial(nrpBi)/factorial(nrpBj)
                            for krpB in range(min(nrpBi,nrpBj)+1):
                                SrmB = 0
                                armB = 1/factorial(nrmBi)/factorial(nrmBj)
                                for krmB in range(min(nrmBi,nrmBj)+1):
                                    pPrime = nsum-2*(kRpA+kRmA+krpA+krmA+kRpB+kRmB+krpB+krmB)
                                    if pPrime+1 < 0:
                                        print('hit!')
                                        print('nSum:', nsum)
                                        print('kSum:', kRpA+kRmA+krpA+krmA+kRpB+kRmB+krpB+krmB)
                                    IPrime = (1/4)**(pPrime+1) * math.gamma(pPrime+1)*mp.hyperu((pPrime+1)/2, (1/2), (1/2)*(dTilde)**2)
                                    SrmB += armB*IPrime
                                    armB *= -1*(nrmBi-krmB)*(nrmBj-krmB)/(krmB+1)
                                SrpB += arpB*SrmB
                                arpB *= -1*(nrpBi-krpB)*(nrpBj-krpB)/(krpB+1)
                            SRmB += aRmB*SrpB
                            aRmB *= -1*(nRmBi-kRmB)*(nRmBj-kRmB)/(kRmB+1)
                        SRpB += aRpB*SRmB
                        aRpB *= -1*(nRpBi-kRpB)*(nRpBj-kRpB)/(kRpB+1)
                    SrmA += armA*SRpB
                    armA *= -1*(nrmAi-krmA)*(nrmAj-krmA)/(krmA+1)
                SrpA += arpA*SrmA
                arpA *= -1*(nrpAi-krpA)*(nrpAj-krpA)/(krpA+1)
            SRmA += aRmA*SrpA
            aRmA *= -1*(nRmAi-kRmA)*(nRmAj-kRmA)/(kRmA+1)
        SRpA += aRpA*SRmA
        aRpA *= -1*(nRpAi-kRpA)*(nRpAj-kRpA)/(kRpA+1)
    L = sqrt(hbar**2/(omgh*mStar*electronMass))
    E0 = eSquaredOvere0/(dielectricConstant*L)
    deltaNRB = nRmBi + nRpBi - nRmBj - nRpBj
    deltaNRA = nRmAi + nRpAi - nRmAj - nRpAj
    deltaNrB = nrmBi + nrpBi - nrmBj - nrpBj
    deltaNrA = nrmAi + nrpAi - nrmAj - nrpAj
    V = SRpA*2**(7/2)*E0*(-1)**(abs(deltaNRB+nRpAj+nRmAj+nrpAj+nrmAj+nRpBj+nRmBj+nrmBj+nrpBj))*1j**(abs(deltaNRB+deltaNRA+deltaNrB+deltaNrA))*sqrt(factorial(nrpAi)*factorial(nrpAj)*factorial(nrmAi)*factorial(nrmAj)*factorial(nrpBi)*factorial(nrpBj)*factorial(nrmBi)*factorial(nrmBj))
    return(V)

def Coul_hh_4body_noSeparation(omgh,nRpAi,nRmAi,nrpAi,nrmAi,nRpAj,nRmAj,nrpAj,nrmAj,nRpBi,nRmBi,nrpBi,nrmBi,nRpBj,nRmBj,nrpBj,nrmBj,mStar,dielectricConstant):
    nsum = nRpAi+nRmAi+nrpAi+nrmAi+nRpAj+nRmAj+nrpAj+nrmAj+nRpBi+nRmBi+nrpBi+nrmBi+nRpBj+nRmBj+nrpBj+nrmBj
    dl = (nRpAj-nRpAi)-(nRmAj-nRmAi)+(nrpAj-nrpAi)-(nrmAj-nrmAi)+(nRpBj-nRpBi)-(nRmBj-nRmBi)+(nrpBj-nrpBi)-(nrmBj-nrmBi)
    if dl !=0:
        return 0
    dlrA = (nrpAj-nrpAi)-(nrmAj-nrmAi)
    dlrB = (nrpBj-nrpBi)-(nrmBj-nrmBi)
    if dlrA%2 != 0 or dlrB%2 != 0:
        return 0
    SRpA = 0
    aRpA = 1/factorial(nRpAi)/factorial(nRpAj)
    for kRpA in range(min(nRpAi,nRpAj)+1):
        SRmA = 0
        aRmA = 1/factorial(nRmAi)/factorial(nRmAj)
        for kRmA in range(min(nRmAi,nRmAj)+1):
            SrpA = 0
            arpA = 1/factorial(nrpAi)/factorial(nrpAj)
            for krpA in range(min(nrpAi,nrpAj)+1):
                SrmA = 0
                armA = 1/factorial(nrmAi)/factorial(nrmAj)
                for krmA in range(min(nrmAi,nrmAj)+1):
                    SRpB = 0
                    aRpB = 1/factorial(nRpBi)/factorial(nRpBj)
                    for kRpB in range(min(nRpBi,nRpBj)+1):
                        SRmB = 0
                        aRmB = 1/factorial(nRmBi)/factorial(nRmBj)
                        for kRmB in range(min(nRmBi, nRmBj)+1):
                            SrpB = 0
                            arpB = 1/factorial(nrpBi)/factorial(nrpBj)
                            for krpB in range(min(nrpBi,nrpBj)+1):
                                SrmB = 0
                                armB = 1/factorial(nrmBi)/factorial(nrmBj)
                                for krmB in range(min(nrmBi,nrmBj)+1):
                                    pPrime = nsum-2*(kRpA+kRmA+krpA+krmA+kRpB+kRmB+krpB+krmB)
                                    IPrime = (1/2)**(pPrime+2) * math.gamma((pPrime+1)/2)
                                    SrmB += armB*IPrime
                                    armB *= -1*(nrmBi-krmB)*(nrmBj-krmB)/(krmB+1)
                                SrpB += arpB*SrmB
                                arpB *= -1*(nrpBi-krpB)*(nrpBj-krpB)/(krpB+1)
                            SRmB += aRmB*SrpB
                            aRmB *= -1*(nRmBi-kRmB)*(nRmBj-kRmB)/(kRmB+1)
                        SRpB += aRpB*SRmB
                        aRpB *= -1*(nRpBi-kRpB)*(nRpBj-kRpB)/(kRpB+1)
                    SrmA += armA*SRpB
                    armA *= -1*(nrmAi-krmA)*(nrmAj-krmA)/(krmA+1)
                SrpA += arpA*SrmA
                arpA *= -1*(nrpAi-krpA)*(nrpAj-krpA)/(krpA+1)
            SRmA += aRmA*SrpA
            aRmA *= -1*(nRmAi-kRmA)*(nRmAj-kRmA)/(kRmA+1)
        SRpA += aRpA*SRmA
        aRpA *= -1*(nRpAi-kRpA)*(nRpAj-kRpA)/(kRpA+1)
    L = sqrt(hbar**2/(omgh*mStar*electronMass))
    E0 = eSquaredOvere0/(dielectricConstant*L)
    deltaNRB = nRmBi + nRpBi - nRmBj - nRpBj
    deltaNRA = nRmAi + nRpAi - nRmAj - nRpAj
    deltaNrB = nrmBi + nrpBi - nrmBj - nrpBj
    deltaNrA = nrmAi + nrpAi - nrmAj - nrpAj
    V = SRpA*2**(7/2)*E0*(-1)**(abs(deltaNRB+nRpAj+nRmAj+nrpAj+nrmAj+nRpBj+nRmBj+nrmBj+nrpBj))*1j**(abs(deltaNRB+deltaNRA+deltaNrB+deltaNrA))*sqrt(factorial(nrpAi)*factorial(nrpAj)*factorial(nrmAi)*factorial(nrmAj)*factorial(nrpBi)*factorial(nrpBj)*factorial(nrmBi)*factorial(nrmBj))
    return(V)

@numba.jit()
def Coul_mat_hh(omgh,states,mStar,dielectricConstant):
    nRp = states[:,0]
    nRm = states[:,1]
    nrp = states[:,2]
    nrm = states[:,3]
    N = len(nRp)
    V = zeros((N,N))
    for i in range(N):
        nRpi = nRp[i]
        nRmi = nRm[i]
        nrpi = nrp[i]
        nrmi = nrm[i]
        for j in range(i+1):
            nRpj = nRp[j]
            nRmj = nRm[j]
            nrpj = nrp[j]
            nrmj = nrm[j]
            if nRpi != nRpj or nRmi != nRmj:
                Vij = 0
            else:
                Vij = Coul_hh_2body_noSeparation(omgh,nRpi,nRmi,nrpi,nrmi,nRpj,nRmj,nrpj,nrmj,mStar,dielectricConstant)
            V[i,j] = Vij
            V[j,i] = Vij
    return V

def ED_hh(E0_s, E0_a, stateList_s, stateList_a, omgh, mStar,dielectricConstant):
    #Given a symmetric basis, diagonalize both in the symmetric basis and the corresponding antisymmetric basis
    Ns = len(E0_s) #number of direct product states that will contributed to the symmetrized basis
    Es_s = zeros(Ns) #energy eigenvalues in the symmetric basis
    evecs_s = zeros((Ns,Ns)) #eigenvectors
    Na = len(E0_a)
    Es_a = zeros(Na)
    evecs_a = zeros((Na,Na))
    nRp_s = stateList_s[:,0]
    nRm_s = stateList_s[:,1]
    nrp_s = stateList_s[:,2]
    nrm_s = stateList_s[:,3]
    nRp_a = stateList_a[:,0]
    nRm_a = stateList_a[:,1]
    nrp_a = stateList_a[:,2]
    nrm_a = stateList_a[:,3]
    ls = nRp_s+nrp_s-nRm_s-nrm_s #angular momentum. Aidan: once again, a 1d array whose i^th entry is the angular momentum of the i^th state
    la = nRp_a+nrp_a-nRm_a-nrm_a
    for l in range(min(ls),max(ls)+1): #diagonalize the block with angular momentum l
        ind_ls = (ls==l) #a 1d array of True/False depending on whether the correpsonding entry in ls=l
        ind_la = (la==l)
        Nls = sum(ind_ls) # number of states in the symmetricCoul_mat_hh direct product basis whose angular momenta = l
        Nla = sum(ind_la)
        if Nls>0:
            E0_ls = E0_s[ind_ls]
            Ham_ls = diag(E0_ls) #noninteracting Hamiltonian
            states_ls = stateList_s[ind_ls]
            Vhh_ls = Coul_mat_hh(omgh,states_ls,mStar,dielectricConstant)
            Ham_ls += Vhh_ls #full Hamiltonian
            energies,evs = linalg.eigh(Ham_ls)
            Es_s[ind_ls] = energies
            evecs_s[ix_(ind_ls,ind_ls)] = evs #the columns of evecs_s are the eigenvectors
        if Nla>0:
            E0_la = E0_a[ind_la]
            Ham_la = diag(E0_la)
            states_la = stateList_a[ind_la]
            Ham_la += Coul_mat_hh(omgh,states_la, mStar,dielectricConstant)
            energies,evs = linalg.eigh(Ham_la)
            Es_a[ind_la] = energies
            evecs_a[ix_(ind_la,ind_la)] = evs
    ind_s = argsort(Es_s) #sort by energy
    Es_s = Es_s[ind_s]
    evecs_s = evecs_s[:,ind_s]
    ls = ls[ind_s]
    ind_a = argsort(Es_a)
    Es_a = Es_a[ind_a]
    evecs_a = evecs_a[:,ind_a]
    la = la[ind_a]
    Es_unsorted = concatenate((Es_s,Es_a))
    ind_sort = argsort(Es_unsorted)
    Es = Es_unsorted[ind_sort]
    parityList = concatenate((zeros(Ns), ones(Na)))[ind_sort]
    return Es, Es_s, evecs_s, Es_a, evecs_a, parityList.astype(int)

#@numba.jit()
def Coul_4body_estateestate(dTilde, sTilde, eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, cutoff, omgh, parityList, basisStates_s, basisStates_a, evecs_s, evecs_a, mStar,dielectricConstant):
    V = 0
    # the issues is that I have a list of all the energy eigenvalues Es, but separates symmetric and antisymmetric evec arrays. So, I need a way to map from my energy eigenstateIndex to its corresponding index in either evec_s or evec_a
    collectiveEStateIndexToAsymIndexMap = cumsum(parityList).astype(int)
    collectiveEStateIndexToSymIndexMap = abs(cumsum(-(parityList-1))).astype(int)
    collectiveEStateIndexToSymAndAsymIndexMap = [collectiveEStateIndexToSymIndexMap, collectiveEStateIndexToAsymIndexMap] # this guy's first column maps Es to evecs_s and its second column maps Es to evecs_a
    AiParity = parityList[eigenstateAiIndex]
    AjParity = parityList[eigenstateAjIndex]
    BiParity = parityList[eigenstateBiIndex]
    BjParity = parityList[eigenstateBjIndex]
    # we have the indices of the eigenstates with respect to Es. Now we want their proper indices, which are the indices within the list of states of their respective exchange symmetry
    eigenstateAiIndexProper = collectiveEStateIndexToSymAndAsymIndexMap[AiParity][eigenstateAiIndex]
    eigenstateAjIndexProper = collectiveEStateIndexToSymAndAsymIndexMap[AjParity][eigenstateAjIndex]
    eigenstateBiIndexProper = collectiveEStateIndexToSymAndAsymIndexMap[BiParity][eigenstateBiIndex]
    eigenstateBjIndexProper = collectiveEStateIndexToSymAndAsymIndexMap[BjParity][eigenstateBjIndex]
    evecs_stacked = [evecs_s, evecs_a]
    basisStates_stacked = [basisStates_s, basisStates_a]
    numBasisStates_stacked = [shape(basisStates_s)[0], shape(basisStates_a)[0]]
    eigenstateAi = evecs_stacked[AiParity][:, eigenstateAiIndexProper]
    basisStatesAi = basisStates_stacked[AiParity]
    numBasisStatesAi=numBasisStates_stacked[AiParity]
    eigenstateAj = evecs_stacked[AjParity][:, eigenstateAjIndexProper]
    basisStatesAj = basisStates_stacked[AjParity]
    numBasisStatesAj=numBasisStates_stacked[AjParity]
    eigenstateBi = evecs_stacked[BiParity][:, eigenstateBiIndexProper]
    basisStatesBi = basisStates_stacked[BiParity]
    numBasisStatesBi=numBasisStates_stacked[BiParity]
    eigenstateBj = evecs_stacked[BjParity][:, eigenstateBjIndexProper]
    basisStatesBj = basisStates_stacked[BjParity]
    numBasisStatesBj=numBasisStates_stacked[BjParity]
    for basisStateAiIndex in range(numBasisStatesAi):
        basisStateAi = basisStatesAi[basisStateAiIndex]
        basisStateAiCoefficient = eigenstateAi[basisStateAiIndex]
        if basisStateAiCoefficient > cutoff:
            nRpAi=basisStateAi[0]
            nRmAi=basisStateAi[1]
            nrpAi=basisStateAi[2]
            nrmAi=basisStateAi[3]
            for basisStateAjIndex in range(numBasisStatesAj):
                basisStateAj = basisStatesAj[basisStateAjIndex]
                basisStateAjCoefficient = eigenstateAj[basisStateAjIndex]
                if basisStateAjCoefficient > cutoff:
                    nRpAj=basisStateAj[0]
                    nRmAj=basisStateAj[1]
                    nrpAj=basisStateAj[2]
                    nrmAj=basisStateAj[3]
                    for basisStateBiIndex in range(numBasisStatesBi):
                        basisStateBi = basisStatesBi[basisStateBiIndex]
                        basisStateBiCoefficient = eigenstateBi[basisStateBiIndex]
                        if basisStateBiCoefficient > cutoff:
                            nRpBi=basisStateBi[0]
                            nRmBi=basisStateBi[1]
                            nrpBi=basisStateBi[2]
                            nrmBi=basisStateBi[3]
                            for basisStateBjIndex in range(numBasisStatesBj):
                                basisStateBj = basisStatesBj[basisStateBjIndex]
                                basisStateBjCoefficient = eigenstateBj[basisStateBjIndex]
                                if basisStateBjCoefficient > cutoff:
                                    nRpBj=basisStateBj[0]
                                    nRmBj=basisStateBj[1]
                                    nrpBj=basisStateBj[2]
                                    nrmBj=basisStateBj[3]
                                    amplitude = basisStateAiCoefficient * basisStateBiCoefficient * conj(basisStateAjCoefficient) * conj(basisStateBjCoefficient)
                                    matElt = Coul_hh_4body_generalSeparation(dTilde,sTilde,omgh,nRpAi,nRmAi,nrpAi,nrmAi,nRpAj,nRmAj,nrpAj,nrmAj,nRpBi,nRmBi,nrpBi,nrmBi,nRpBj,nRmBj,nrpBj,nrmBj,mStar,dielectricConstant)
                                    V+=amplitude*matElt
    return(V)

"""
@njit
def computeFormFactor(q,qPrime,omgh,npi,nmi,npj,nmj,dielectricConstant=dielectricConstant):
    nsum = npj+nmj+nmi+npi
    dl = (npj-nmj)-(npi-nmi)
    Sp = 0
    ap = 1/factorial(npi)/factorial(npj)
    for kp in range(min(npi,npj)+1):
        Sm = 0
        am = 1/factorial(nmi)/factorial(nmj)
        for km in range(min(nmi,nmj)+1):
            p = nsum-2*(kp+km)
            Sm += am*abs(q)**(nmj+nmi-2km)*abs(qPrime)**(npj+npi-2kp)
            am *= -1*(nmi-km)*(nmj-km)/(km+1)
        Sp += ap*Sm
        ap *= -1*(npi-kp)*(npj-kp)/(kp+1)
    L = sqrt(hbar**2/(omgh*m))
    E0 = eSquaredOvere0/(dielectricConstant*L) # this is in SI and is equivalent to 2*pi*e^2/(eps*L) in CGS ##CHECK FOR CONSISTENCY/ACCURACY OF UNITS
    Vij = ((-1)**(npj+nmj))*2*E0*Srp*sqrt(factorial(npi)*factorial(npj)*factorial(nmi)*factorial(nmj))*((npj-npi)+(nmj-nmi))
    return(V)


check = Coul_hh_generalSeparation(dTilde,sTilde,omgh,nRpAi,nRmAi,nrpAi,nrmAi,nRpAj,nRmAj,nrpAj,nrmAj,nRpBi,nRmBi,nrpBi,nrmBi,nRpBj,nRmBj,nrpBj,nrmBj,dielectricConstant=dielectricConstant)
test = Coul_hh_generalSeparation(dTilde,sTilde,omgh,nRpBj,nRmBj,nrpBj,nrmBj,nRpBi,nRmBi,nrpBi,nrmBi,nRpAj,nRmAj,nrpAj,nrmAj,nRpAi,nRmAi,nrpAi,nrmAi,dielectricConstant=dielectricConstant)
if isclose(check, test) == True:
  print('yay!')
if isclose(check, test) == False:
    print('hit!')
"""

"""
 if pPrime+1 < 0:
                                        print('hit!')
                                        print('nSum:', nsum)
                                        print('kSum:', kRpA+kRmA+krpA+krmA+kRpB+kRmB+krpB+krmB)
                                        print(nRpAi, nRpAj, kRpA)
                                        print(nRmAi, nRmAj, kRmA)
                                        print(nrpAi, nrpAj, krpA)
                                        print(nrmAi, nrmAj, krmA)
                                        print(nRpBi, nRpBj, kRpB)
                                        print(nRmBi, nRmBj, kRmB)
                                        print(nrpBi, nrpBj, krpB)
                                        print(nrmBi, nrmBj, krmB)
"""
