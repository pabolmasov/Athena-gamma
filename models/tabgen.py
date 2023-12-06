from numpy import *
from matplotlib import gridspec

import os
import sys
import glob
sys.path.append("/Users/pasha/athena/vis/python")
import athena_read

import gc

ifplot = True
if(ifplot):
    import matplotlib
    from matplotlib.pyplot import *

# softenedpolytropic([-8.0, 1.0], [-10.0,10.0], -8, -7., 5./3., nrho=10)

def softenedpolytropic(rhorange, erange, rhobreak1, rhobreak2, gamma, nrho = 10, nen = 2):

    rhoar = (arange(nrho)/double(nrho-1)) * (rhorange[1]-rhorange[0])+rhorange[0]

    # adiabatic index depending on density:
    gamma_min = 1.001
    gamma1 = ones(nrho)
    gamma1 = (gamma-gamma_min) * (rhoar-rhobreak1)/(rhobreak2-rhobreak1)+gamma_min
    gamma1 = minimum(maximum(gamma1, gamma_min), gamma)
    
    ear = (arange(nen)/double(nen-1)) * (erange[1]-erange[0])+erange[0]

    f = open('e.tab', 'w')
    
    f.write('# Entries must be space separated.\n')
    f.write('#  n_var,  n_espec, n_rho\n')
    f.write('# (fields) (rows)  (columns)\n')
    f.write('4 '+str(nen)+' '+str(nrho)+'\n')
    f.write('# Log espec lim (specific internal energy e/rho)\n')
    f.write(f'{erange[0]:.4e} {erange[1]:.4e}\n')
    f.write('# Log rho lim\n')
    f.write(f'{rhorange[0]:.4e} {rhorange[1]:.4e}\n')
    f.write('# Ratios = 1, eint/pres, eint/pres, eint/h\n')
    f.write('# This line is required iff EOS_read_ratios\n')
    print(f'{1.:.4e} {1./(gamma-1.):.4e} {1./(gamma-1.):.4e}  {1./gamma:.4e}\n')
    f.write(f'{1.:.4e} {1./(gamma-1.):.4e} {1./(gamma-1.):.4e}  {1./gamma:.4e}\n')
    f.write('# Log p/e(e/rho,rho)\n')
    for ken in arange(nen):
        for k in arange(nrho):
            f.write(f'{log10(gamma1[k]-1.):.4e}')
            if k<(nrho-1):
                f.write(' ')
        f.write('\n')
    f.write('# Log e/p(e/rho,rho)\n')
    for ken in arange(nen):
        for k in arange(nrho):
            f.write(f'{-log10(gamma1[k]-1.):.4e}')
            if k<(nrho-1):
                f.write(' ')
        f.write('\n')
    f.write('# Log asq*rho/p(p/rho,rho)\n')
    for ken in arange(nen):
        for k in arange(nrho):
            f.write(f'{log10(gamma1[k]):.4e}')
            if k<(nrho-1):
                f.write(' ')
        f.write('\n')

    f.flush()
    f.close()
