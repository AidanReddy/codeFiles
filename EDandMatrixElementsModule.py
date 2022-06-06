from numpy import *
import scipy.linalg as la
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
electronMass = 5.856301 * 10**(-29) # meV *(second/Å)
eSquaredOvere0 =  14400 #meV * angstrom #CGS

def oneBody_basis(N, omgh):
    numBasisStates = int((N+1)*(N+2)/2)
    npList = zeros(numBasisStates)
    nmList = zeros(numBasisStates)
    index = 0
    for np in range(N+1):
        for nm in range(N-np+1):
            npList[index] = np
            nmList[index] = nm
            index += 1
    E0 = (npList+nmList)*omgh
    ind = argsort(E0)
    E0 = E0[ind] + omgh
    npList = npList[ind]
    nmList = nmList[ind]
    return(E0,npList.astype(int),nmList.astype(int))

def nonint_basis_symmetric(N, omgh):
    numBasisStates = int((1/24)*((N+1)*(N+2)*(N+3)*(N+4))) # the number of 2 particle 2DIHO states at or below the the N^th noninteracting energy level
    #First construct a basis with size (nhp*nhm)**2, then pick out Ncut lowest states (Ncut is the number of two-particle basis states we retain)
    nRpList = zeros(numBasisStates)
    nRmList = zeros(numBasisStates)
    nrpList = zeros(numBasisStates)
    nrmList = zeros(numBasisStates)
    index = 0
    for nRp in range(N+1):
        for nRm in range(N-nRp+1):
            for nrp in range(N-nRp-nRm+1):
                for nrm in range(N-nRp-nRm-nrp+1):
                    nRpList[index] = nRp
                    nRmList[index] = nRm
                    nrpList[index] = nrp
                    nrmList[index] = nrm
                    index += 1
    ind = abs(nrpList-nrmList)%2 == 0
    nRpList = nRpList[ind]
    nRmList = nRmList[ind]
    nrpList = nrpList[ind]
    nrmList = nrmList[ind]
    E0 = (nRpList+nRmList+nrpList+nrmList)*omgh
    ind = argsort(E0)
    E0_s = E0[ind] + 2*omgh
    nRpList_s = nRpList[ind]
    nRmList_s = nRmList[ind]
    nrpList_s = nrpList[ind]
    nrmList_s = nrmList[ind]
    return(E0_s,nRpList_s.astype(int),nRmList_s.astype(int),nrpList_s.astype(int),nrmList_s.astype(int))

def nonint_basis_antisymmetric(N, omgh):
    numBasisStates = int((1/24)*((N+1)*(N+2)*(N+3)*(N+4))) # the number of 2 particle 2DIHO states at or below the the N^th noninteracting energy level
    #First construct a basis with size (nhp*nhm)**2, then pick out Ncut lowest states (Ncut is the number of two-particle basis states we retain states)
    nRpList = zeros(numBasisStates)
    nRmList = zeros(numBasisStates)
    nrpList = zeros(numBasisStates)
    nrmList = zeros(numBasisStates)
    index = 0
    for nRp in range(N+1):
        for nRm in range(N-nRp+1):
            for nrp in range(N-nRp-nRm+1):
                for nrm in range(N-nRp-nRm-nrp+1):
                    nRpList[index] = nRp
                    nRmList[index] = nRm
                    nrpList[index] = nrp
                    nrmList[index] = nrm
                    index += 1
    ind = abs(nrpList-nrmList)%2 == 1
    nRpList = nRpList[ind]
    nRmList = nRmList[ind]
    nrpList = nrpList[ind]
    nrmList = nrmList[ind]
    E0 = (nRpList+nRmList+nrpList+nrmList)*omgh
    ind = argsort(E0)
    E0_a = E0[ind] + 2*omgh
    nRpList_a = nRpList[ind]
    nRmList_a = nRmList[ind]
    nrpList_a = nrpList[ind]
    nrmList_a = nrmList[ind]
    return(E0_a,nRpList_a.astype(int),nRmList_a.astype(int),nrpList_a.astype(int),nrmList_a.astype(int))

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
def Coul_mat_hh(omgh,basis,mStar,dielectricConstant):
    E0,nRp,nRm,nrp,nrm = basis
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

def ED_hh(basis_s, basis_a, omgh, mStar,dielectricConstant):
    #Given a symmetric basis, diagonalize both in the symmetric basis and the corresponding antisymmetric basis
    E0_s,nRp_s,nRm_s,nrp_s,nrm_s = basis_s
    E0_a,nRp_a,nRm_a,nrp_a,nrm_a = basis_a
    Ns = len(E0_s) #number of direct product states that will contributed to the symmetrized basis
    Es_s = zeros(Ns) #energy eigenvalues in the symmetric basis
    evecs_s = zeros((Ns,Ns)) #eigenvectors
    Na = len(E0_a)
    Es_a = zeros(Na)
    evecs_a = zeros((Na,Na))
    ls = nRp_s+nrp_s-nRm_s-nrm_s #angular momentum. Aidan: once again, a 1d array whose i^th entry is the angular momentum of the i^th state
    ls_a = nRp_a+nrp_a-nRm_a-nrm_a
    for l in range(min(ls),max(ls)+1): #diagonalize the block with angular momentum l
        ind_ls = (ls==l) #a 1d array of True/False depending on whether the correpsonding entry in ls=l
        ind_la = (ls_a==l)
        Nl_s = sum(ind_ls) # number of states in the symmetricCoul_mat_hh direct product basis whose angular momenta = l
        Nl_a = sum(ind_la)
        if Nl_s>0:
            E0_ls = E0_s[ind_ls]
            Ham_ls = diag(E0_ls) #noninteracting Hamiltonian
            nRm_ls = nRm_s[ind_ls]
            nRp_ls = nRp_s[ind_ls]
            nrm_ls = nrm_s[ind_ls]
            nrp_ls = nrp_s[ind_ls]
            basis_ls = (E0_ls,nRp_ls,nRm_ls,nrp_ls,nrm_ls)
            Vhh_ls = Coul_mat_hh(omgh,basis_ls,mStar,dielectricConstant)
            Ham_ls += Vhh_ls #full Hamiltonian
            energies,evs = la.eigh(Ham_ls)
            Es_s[ind_ls] = energies
            evecs_s[ix_(ind_ls,ind_ls)] = evs #the columns of evecs_s are the eigenvectors
        if Nl_a>0:
            E0_la = E0_a[ind_la]
            Ham_la = diag(E0_la)
            nRp_la = nRp_a[ind_la]
            nRm_la = nRm_a[ind_la]
            nrp_la = nrp_a[ind_la]
            nrm_la = nrm_a[ind_la]
            basis_la = (E0_la,nRp_la,nRm_la,nrp_la,nrm_la)
            Ham_la += Coul_mat_hh(omgh,basis_la,mStar,dielectricConstant)
            energies,evs = la.eigh(Ham_la)
            Es_a[ind_la] = energies
            evecs_a[ix_(ind_la,ind_la)] = evs
    ind = argsort(Es_s) #sort by energy
    Es_s = Es_s[ind]
    evecs_s = evecs_s[:,ind]
    ls_s = ls[ind]
    ind = argsort(Es_a)
    Es_a = Es_a[ind]
    evecs_a = evecs_a[:,ind]
    ls_a = ls_a[ind]
    return Es_s, evecs_s, Es_a, evecs_a

#function to calculate flip flop Coulomb interaction between s=two atoms. Flip flop means that the atoms switch states, i.e. <eigenstate beta|<eigenstate alpha|V|eigenstate alpha>|eigenstate beta>. In other words, only two unique eigenstates involved, as opposed to four.
#@numba.jit()
def Coul_4body_estateestate(dTilde, sTilde, eigenstateAiIndex, eigenstateAjIndex, eigenstateBiIndex, eigenstateBjIndex, cutoff, omgh, basisStatesMatrix, evecs, mStar,dielectricConstant):
    V = 0
    numBasisStates = shape(basisStatesMatrix)[1]
    eigenstateAi = evecs[:, eigenstateAiIndex]
    eigenstateAj = evecs[:, eigenstateAjIndex]
    eigenstateBi= evecs[:, eigenstateBiIndex]
    eigenstateBj= evecs[:, eigenstateBjIndex]
    for basisStateAiIndex in range(numBasisStates):
        basisStateAi = basisStatesMatrix[:, basisStateAiIndex]
        basisStateAiCoefficient = eigenstateAi[basisStateAiIndex]
        if basisStateAiCoefficient > cutoff:
            nRpAi=basisStateAi[0]
            nRmAi=basisStateAi[1]
            nrpAi=basisStateAi[2]
            nrmAi=basisStateAi[3]
            for basisStateAjIndex in range(numBasisStates):
                basisStateAj = basisStatesMatrix[:, basisStateAjIndex]
                basisStateAjCoefficient = eigenstateAj[basisStateAjIndex]
                if basisStateAjCoefficient > cutoff:
                    nRpAj=basisStateAj[0]
                    nRmAj=basisStateAj[1]
                    nrpAj=basisStateAj[2]
                    nrmAj=basisStateAj[3]
                    for basisStateBiIndex in range(numBasisStates):
                        basisStateBi = basisStatesMatrix[:, basisStateBiIndex]
                        basisStateBiCoefficient = eigenstateBi[basisStateBiIndex]
                        if basisStateBiCoefficient > cutoff:
                            nRpBi=basisStateBi[0]
                            nRmBi=basisStateBi[1]
                            nrpBi=basisStateBi[2]
                            nrmBi=basisStateBi[3]
                            for basisStateBjIndex in range(numBasisStates):
                                basisStateBj = basisStatesMatrix[:, basisStateBjIndex]
                                basisStateBjCoefficient = eigenstateBj[basisStateBjIndex]
                                if basisStateBjCoefficient > cutoff:
                                    nRpBj=basisStateBj[0]
                                    nRmBj=basisStateBj[1]
                                    nrpBj=basisStateBj[2]
                                    nrmBj=basisStateBj[3]
                                    amplitude = basisStateAiCoefficient * basisStateBiCoefficient * conj(basisStateAjCoefficient) * conj(basisStateBjCoefficient)
                                    V+=amplitude*Coul_hh_4body_generalSeparation(dTilde,sTilde,omgh,nRpAi,nRmAi,nrpAi,nrmAi,nRpAj,nRmAj,nrpAj,nrmAj,nRpBi,nRmBi,nrpBi,nrmBi,nRpBj,nRmBj,nrpBj,nrmBj,mStar,dielectricConstant)
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
