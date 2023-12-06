import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from numpy import *
import numpy.ma as ma
from pylab import *
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import minimize, root, root_scalar
import glob
import re
import os

from cmath import phase

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

close('all')
ioff()
use('Agg')

ncoeff = 1.5

dx = 1e-5

# rhoc = 1.
rstar = 1.
mstar = 1.
rhoc = 1.
x=0.

theta = 1. # conditions in the center
dtheta = 0. #

dxout = 0.001
xstore = 0.

amode = 1. # the a factor depedent on theta (to be set later)

rrshape = 0.
rsign = 1.

xlist = []  ; thlist = []

while((theta>0.)&(x<1000.)):
    ddtheta = -theta**ncoeff * x**2
    dtheta += ddtheta * dx/2.
    bsq = theta**((2.*ncoeff+1.)) * x**2
    rrshape += rsign * sqrt(maximum(bsq-(amode*rrshape/maximum(x, dx))**2, 0.)) * dx/2.

    theta += dtheta/maximum(x,dx)**2 * dx
    x += dx
    ddtheta = -theta**ncoeff * x**2
    dtheta += ddtheta * dx/2.
    #
    if (x>xstore):
        xlist.append(x)
        thlist.append(theta)
        xstore += dxout
        # print(x, theta)
        
# renormalisation!
xlist = asarray(xlist)
thlist = asarray(thlist)

clf()
plot(xlist, thlist, 'k.')
# plot(xlist, sin(xlist)/maximum(xlist, dx), 'r-')
plot(xlist, 1./sqrt(xlist**2/3.+1.), 'r-')
xlabel(r'$\xi$')
ylabel(r'$\theta$')
savefig('laplot.png')

xlast = xlist[-1]

alpha = rstar / xlast
# {\displaystyle \alpha ^{2}=(n+1)K\rho _{c}^{{\frac {1}{n}}-1}/4\pi G,}
pc =4.*pi /(ncoeff+1.)
# rhoc**(1.+1./ncoeff) * alpha**2/(ncoeff+1.)/rhoc**(1./ncoeff-1.)*4.*pi

rho = rhoc * thlist**ncoeff
P = pc * thlist**(1.+ncoeff)

r = xlist 

# rlist *= sqrt(pc*rhoc) * alpha

mass = cumtrapz(rho*4.*pi*r**2, x=r, initial=0.)

mfactor = mstar / mass[-1]

rfactor = rstar / xlast

# rfactor = 1.

mass *= mfactor ; rho *= mfactor / rfactor**3 ; P *= (mfactor/rfactor**2)**2

r *= rfactor

fout = open('emden.dat', 'w')

for k in arange(size(xlist)):
    kr = size(xlist)-1-k
    fout.write(str(mass[kr])+' '+str(r[kr])+' '+str(rho[kr])+' '+str(P[kr])+'\n')
    
fout.flush()
fout.close()

# mass check:
mtot = trapz(rho * 4. * pi * r**2, x=r)

print("M = ", mtot)
print(rfactor, mfactor)
