import numpy as np
import scipy.linalg as la
import numpy.linalg #I also import this because numba support numpy linalg but not necessarily scipy linalg functions
import math
import timeit
from numba import njit
from numba import jit

electronMass = 5.685 * 10**(-29) # meV *(second/Å)^2
hbar =  6.582 * 10**(-13) # meV * s

def computeShell(n, am): # working -3/24/22 - solved issues that were arising when theta = 3
    b1Unitless = (4*math.pi/math.sqrt(3)) * np.array([1,0]) # define reciprocal basis vectors b1 and b2 for moire lattice
    b2Unitless = (4*math.pi/math.sqrt(3)) * np.array([0.5, math.sqrt(3)/2])
    shell = np.array([])
    gCount = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            g = i * b1Unitless + j * b2Unitless
            if la.norm(g) <= la.norm(n*b1Unitless):
                if i + j <= n:
                    if -(j + i) <= n:
                        shell = np.append(shell, g)
                        gCount += 1
    shell = shell.reshape(gCount, 2)
    shell = np.delete(shell, gCount//2, 0)
    scrapShell = shell
    normArray = np.array([])
    for i in range(gCount-1):
        normArray = np.append(normArray, la.norm(scrapShell[i]))
    finalShellNorms = np.unique(normArray)
    newFinalShellNorms = np.array([])
    for i in finalShellNorms:
        redundant = False
        for j in newFinalShellNorms:
            if np.allclose(i,j):
                redundant = True
        if redundant == False:
            newFinalShellNorms = np.append(newFinalShellNorms, i)
    finalShellNorms = newFinalShellNorms
    finalShell = np.array([])
    finalShellCount = 0
    for i in range(n):
        normVal = finalShellNorms[i]
        for j in shell:
            if np.allclose(la.norm(j), normVal):
                finalShellCount += 1
                finalShell = np.append(finalShell, j)
    finalShell = np.append(finalShell, [0,0])
    finalShell = finalShell.reshape(finalShellCount+1, 2)
    finalShell = np.roll(finalShell, 1, 0) # this function shifts every row index down by one so that the last row, which is (0,0), ends up first
    finalShell = finalShell/am
    return(finalShell)


#hexagonal mesh, with symmetry reducedMesh
def computeMesh(N, am): # working -3/24/22 - solved issues that were arising when theta = 3
    b1Unitless = 4*np.pi/np.sqrt(3) * np.array([1,0])
    b2Unitless = (4*np.pi/np.sqrt(3)) * np.array([0.5, np.sqrt(3)/2])
    #generate Monkhorst-Pack grid
    firstShellUnitless = computeShell(1, 1)
    mesh = np.array([])
    for i in range(1, N + 1):
        u1 = (2*i - N - 1)/(2 * N)
        for j in range(1, N + 1):
            u2 = (2*j - N - 1)/(2 * N)
            kVal = u1 * b1Unitless + u2 * b2Unitless
            mesh = np.append(mesh, kVal)
    mesh = mesh.reshape(N**2, 2)
    #move into Brillouin zone
    for kValIndex in range(np.shape(mesh)[0]):
        k = mesh[kValIndex]
        for g in firstShellUnitless:
            if la.norm(k+g) < la.norm(mesh[kValIndex]):
                mesh[kValIndex] = k+g
    #symmetry reduce using D6 symmetry
    #dummy way to initialize a two-columed reduced mesh array, will remove this row later
    reducedMesh = np.array([100,100])
    reducedMeshCounter = np.array([0])
    #meshToReducedMeshIndexMap has ith row entry as the corresponding index in the reducedMesh of the ith element of the full Mesh
    meshToReducedMeshIndexMap = np.zeros(N**2)
    rotTheta = math.pi/3
    refAxes = np.array([[1,0],[math.sqrt(3)/2,1/2], [1/2, math.sqrt(3)/2], [0,1], [-1/2, math.sqrt(3)/2], [-math.sqrt(3)/2, 1/2]])
    for kIndex in range(np.shape(mesh)[0]):
        k = mesh[kIndex]
        matched = False
        while matched == False:
            for i in range(0,7):
                rotMat = np.array([[math.cos(i*rotTheta),-math.sin(i*rotTheta)],[math.sin(i*rotTheta),math.cos(i*rotTheta)]])
                kRot = rotMat@k
                for kReducedIndex in range(np.shape(reducedMesh)[0]):
                    kReduced = reducedMesh[kReducedIndex]
                    if np.allclose(kRot, kReduced) and matched == False:
                        meshToReducedMeshIndexMap[kIndex] += kReducedIndex
                        reducedMeshCounter[kReducedIndex] += 1
                        matched = True
            if matched == False:
                for refAxis in refAxes:
                    kRef = -1*k + 2*(k@refAxis)*refAxis
                    for kReducedIndex in range(np.shape(reducedMesh)[0]):
                        kReduced = reducedMesh[kReducedIndex]
                        if np.allclose(kRef, kReduced) and matched == False:
                            meshToReducedMeshIndexMap[kIndex] += kReducedIndex
                            reducedMeshCounter[kReducedIndex] += 1
                            matched = True
            if matched == False:
                reducedMesh = np.vstack([reducedMesh, k])
                reducedMeshCounter = np.append(reducedMeshCounter, np.array([1]))
                meshToReducedMeshIndexMap[kIndex] += np.shape(reducedMeshCounter)[0]-1
                ## matched not really true, but just to get out of while loop
                matched = True
    reducedMesh = np.delete(reducedMesh, 0, 0)
    reducedMeshCounter = np.delete(reducedMeshCounter, 0,0)
    meshToReducedMeshIndexMap -= 1 # since we are removing the zeroth element in the previous two lines
    meshToReducedMeshIndexMap =  meshToReducedMeshIndexMap.astype(int)
    mesh *= 1/am
    reducedMesh *= 1/am
    return(mesh, reducedMesh, reducedMeshCounter, meshToReducedMeshIndexMap)

#kronecker delta
def kronDelta(a,b):
    if np.array_equal(a,b):
        return(1)
    else:
        return(0)

#modulation potential
def computeV(b):
    if b in shells:
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
    matrixElement = -1 * kronDelta(g, gprime) * (hbar)**2 *(1/(2*mStar*electronMass)) * np.dot(k+g, k+g) + V
    return(matrixElement)

def computekVals(direction, points = 100, min = -10, max = 10, startScal = 0, startPoint = np.array([0.0])):
    coefficients = list(np.linspace(min, max, points))
    kVals = np.array([])
    for i in coefficients:
        newKVal = startPoint + (i * direction)
        kVals = np.append(kVals, newKVal)
    kVals = kVals.reshape(points, 2)
    kValsScal = (np.asarray(coefficients)) * np.linalg.norm(direction)/np.linalg.norm(b1) + startScal
    return(kVals, kValsScal)

# compute matrix
def computeMatrix(k):
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

#for k path
def computeTotalEigValMatrix(kVals, gVals):
    totalEigValMatrix = np.array([])
    numKVals = kVals.shape[0]
    for i in range(numKVals):
        k = kVals[i]
        eigVals, eigVecs = computeEigStuff(k)
        totalEigValMatrix = np.append(totalEigValMatrix, eigVals)
    totalEigValMatrix = totalEigValMatrix.reshape(len(kVals), len(gVals))
    totalEigValMatrix = np.transpose(totalEigValMatrix)
    return(totalEigValMatrix)


#for total mesh
def computeMegaEigStuff():
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



##Old


"""
def computeShell(n, am):
    shell = np.array([])
    gCount = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            g = i * b1 + j * b2
            if la.norm(g) <= la.norm(n*b1):
                if i + j <= n:
                    if -(j + i) <= n:
                        shell = np.append(shell, g)
                        gCount += 1
    shell = shell.reshape(gCount, 2)
    shell = np.delete(shell, gCount//2, 0)
    scrapShell = shell
    normArray = np.array([])
    for i in range(gCount-1):
        normArray = np.append(normArray, la.norm(scrapShell[i]))
    finalShellNorms = np.unique(normArray)
    newFinalShellNorms = np.array([])
    for i in finalShellNorms:
        redundant = False
        for j in newFinalShellNorms:
            if np.allclose(i,j):
                redundant = True
        if redundant == False:
            newFinalShellNorms = np.append(newFinalShellNorms, i)
    finalShellNorms = newFinalShellNorms
    finalShell = np.array([])
    finalShellCount = 0
    for i in range(n):
        normVal = finalShellNorms[i]
        for j in shell:
            if np.allclose(la.norm(j), normVal):
                finalShellCount += 1
                finalShell = np.append(finalShell, j)
    finalShell = np.append(finalShell, [0,0])
    finalShell = finalShell.reshape(finalShellCount+1, 2)
    finalShell = np.roll(finalShell, 1, 0) # this function shifts every row index down by one so that the last row, which is (0,0), ends up first
    return(finalShell)
"""

"""
def computeMesh(am):
    #generate Monkhorst-Pack grid
    firstShell = computeShell(1, am)
    mesh = np.array([])
    for i in range(1, N + 1):
        u1 = (2*i - N - 1)/(2 * N)
        for j in range(1, N + 1):
            u2 = (2*j - N - 1)/(2 * N)
            kVal = u1 * b1 + u2 * b2
            mesh = np.append(mesh, kVal)
    mesh = mesh.reshape(N**2, 2)
    #move into Brillouin zone
    for kValIndex in range(np.shape(mesh)[0]):
        k = mesh[kValIndex]
        for g in firstShell:
            if la.norm(k+g) < la.norm(mesh[kValIndex]):
                mesh[kValIndex] = k+g
    #symmetry reduce using D6 symmetry
    #dummy way to initialize a two-columed reduced mesh array, will remove this row later
    reducedMesh = np.array([100,100])
    reducedMeshCounter = np.array([0])
    #meshToReducedMeshIndexMap has ith row entry as the corresponding index in the reducedMesh of the ith element of the full Mesh
    meshToReducedMeshIndexMap = np.zeros(N**2)
    rotTheta = math.pi/3
    refAxes = np.array([[1,0],[math.sqrt(3)/2,1/2], [1/2, math.sqrt(3)/2], [0,1], [-1/2, math.sqrt(3)/2], [-math.sqrt(3)/2, 1/2]])
    for kIndex in range(np.shape(mesh)[0]):
        k = mesh[kIndex]
        matched = False
        while matched == False:
            for i in range(0,7):
                rotMat = np.array([[math.cos(i*rotTheta),-math.sin(i*rotTheta)],[math.sin(i*rotTheta),math.cos(i*rotTheta)]])
                kRot = rotMat@k
                for kReducedIndex in range(np.shape(reducedMesh)[0]):
                    kReduced = reducedMesh[kReducedIndex]
                    if np.allclose(kRot, kReduced) and matched == False:
                        meshToReducedMeshIndexMap[kIndex] += kReducedIndex
                        reducedMeshCounter[kReducedIndex] += 1
                        matched = True
            if matched == False:
                for refAxis in refAxes:
                    kRef = -1*k + 2*(k@refAxis)*refAxis
                    for kReducedIndex in range(np.shape(reducedMesh)[0]):
                        kReduced = reducedMesh[kReducedIndex]
                        if np.allclose(kRef, kReduced) and matched == False:
                            meshToReducedMeshIndexMap[kIndex] += kReducedIndex
                            reducedMeshCounter[kReducedIndex] += 1
                            matched = True
            if matched == False:
                reducedMesh = np.vstack([reducedMesh, k])
                reducedMeshCounter = np.append(reducedMeshCounter, np.array([1]))
                meshToReducedMeshIndexMap[kIndex] += np.shape(reducedMeshCounter)[0]-1
                ## matched not really true, but just to get out of while loop
                matched = True
    reducedMesh = np.delete(reducedMesh, 0, 0)
    reducedMeshCounter = np.delete(reducedMeshCounter, 0,0)
    meshToReducedMeshIndexMap -= 1 # since we are removing the zeroth element in the previous two lines
    meshToReducedMeshIndexMap =  meshToReducedMeshIndexMap.astype(int)
    return(mesh, reducedMesh, reducedMeshCounter, meshToReducedMeshIndexMap)
"""
