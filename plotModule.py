import matplotlib as mpl
import proplot as pplt
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
today = date.today()
date = today.strftime("%b%d%Y")

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (3.375,3.00)
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams["savefig.jpeg_quality"]
#plt.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['text.usetex']= True

figSaveDir='/Users/aidanreddy/Desktop/resonantEnergyTransfer/figures/'

pplt.rc['grid'] = False

def produceEDSpectrumPlot(lambdaVals, megaEigVals_s, megaEigVals_a, numLevelstoPlot, heisenbergEnergies, omgh):
    #just ground state
    fig, ax = plt.subplots()
    ax.plot(lambdaVals, lambdaVals*np.sqrt(np.pi/2)+2, color ='grey', label = 'linear approx.') # first order perturbation theory prediction
    ax.plot(lambdaVals, heisenbergEnergies[1], color = 'blue', label = 'Heisenberg approx.') #heisenberg energy minimization prediction
    ax.plot(lambdaVals, megaEigVals_s[0]/omgh, color = 'black', label = 'exact')
    ax.margins(0)
    ax.legend(frameon=False)
    ax.tick_params(which='both',direction='in')
    ax.set(xlabel=r'$\lambda$', ylabel = r'$E_g/(\hbar\Omega)$', ylim = (2,np.max(megaEigVals_s[0])))
    plt.savefig(figSaveDir+'EDGroundStateSpectrum%s.pdf' % date, bbox_inches='tight')
    plt.show()
    #fullSpectrum
    fig, ax = plt.subplots()
    for ELevelIndex in range(numLevelstoPlot):    
        ax.plot(lambdaVals, (megaEigVals_s[ELevelIndex]-megaEigVals_s[0])/omgh, linewidth='1.2', color = 'black')
    for ELevelIndex in range(numLevelstoPlot):
        ax.plot(lambdaVals, (megaEigVals_a[ELevelIndex]-megaEigVals_s[0])/omgh, linewidth='1.2', color = 'black', linestyle = 'dashed')
    ax.set(xlabel=r'$\lambda$', ylabel = r'$(E-E_{g})/(\hbar\Omega)$', ylim = (-0.2,3.2))
    ax.margins(0)
    ax.tick_params(which='both',direction='in')
    plt.savefig(figSaveDir+'EDSpectrumGap%s.pdf' % date, bbox_inches='tight')
    plt.show()

def produceSingleMoireAtomGLineCutInTPlot(GVals, TVals, theta, dTilde):
    fig, ax = plt.subplots()
    ax.plot(TVals, GVals*10**(-6), color = 'black')
    ax.set(xlabel=r'$T$(K)', ylabel=r'$G$(MWm$^{-2}$K$^{-1}$)')
    ax.margins(0)
    ax.tick_params(which='both',direction='in')
    #plt.savefig(figSaveDir+'moireAtomGTLineCut%stheta=%d' % (date, theta))
    plt.show()