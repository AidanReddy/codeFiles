import numpy as np
from numba import njit

@njit
def materialContinuumModelParameters(material):
    if material == '2DEG':
        a = 0.1 # angstrom
        V1 = 0 # meV
        V2 = 0
        V3 = 0
        phi = 0
        mStar = 1
    if material == 'nearlyFree':
        a = 0.1 # angstrom
        V1 = 0.5 # meV
        V2 = 0
        V3 = 0
        phi = 45 * (np.pi/180)
        mStar = 0.35
    if material == 'WSe2WS2':
        a = 3.283
        V1 = 15
        V2 = 0
        V3 = 0
        phi = 45 * (np.pi/180)
        mStar = 0.42
    if material == 'triangularLattice':
        a = 3.283
        V1 = 15
        V2 = 0
        V3 = 0
        phi = 0 * (np.pi/180)
        mStar = 0.42
    if material == 'atomicLimit':
        a = 10 # angstrom
        V1 = 100 # meV
        V2 = 100
        V3 = 100
        phi = 0
        mStar = 0.1
    if material == 'WSe2MoSe2':
        a = 3.283 
        V1 = 6.6
        V2 = 0
        V3 = 0
        phi = -94 * (np.pi/180)
        mStar = 0.35
    if material =='WS2':
        a = 3.18
        V1 = 33.5
        V2 = 4.0
        V3 = 5.5
        phi = np.pi
        mStar = 0.87
    if material =='MoS2':
        a = 3.182
        V1 = 39.45
        V2 = 6.5
        V3 = 10.0
        phi = np.pi
        mStar = 0.9
    return(a, V1, V2, V3, phi, mStar)