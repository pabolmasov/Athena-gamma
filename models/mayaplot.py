import matplotlib
from matplotlib import rc
import numpy
from numpy import *
import numpy.ma as ma
from pylab import *
from scipy.optimize import root_scalar
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy.signal import *

import h5py

#Uncomment the following if you want to use LaTeX in figures
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 
# plotting:
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager
# from matplotlib.patches import Wedge
from matplotlib import use
# ioff()
# use('Agg')

# for normal distribution fit:
# import matplotlib.mlab as mlab
# from scipy.stats import norm
# for curve fitting:
from scipy.optimize import curve_fit
import os
import sys
sys.path.append("/Users/pasha/athena/vis/python")
import athena_read

from numpy.ma import masked_array

# ifmayavi = True
# if ifmayavi:
import mayavi.mlab as maya
from mlabtex import mlabtex # Sebastian Muller's mlabtex
maya.options.offscreen = False

import gc

tper = 10.0
rper = 20.0
addmass = 1.0e6

def time_from_anomaly(f, dt):
    # Barker's equation
    tf = tan(f/2.)
    return sqrt(2.*rper**3/addmass)*tf*(1.+tf**2/3.) - dt

def BHcoords(time):
    # finding the root of Barker's equation
    f = root_scalar(time_from_anomaly, args=(time-tper), bracket = [-pi*0.99, pi*0.99])
    print("true anomaly ",f)
    r = 2.*rper/(1.+cos(f.root))
    x = r * cos(f.root)
    y = r * sin(f.root)
    
    return x,y

def magplot(nfile, dir = 'M1', ifradial = False, ifmovie=False, ifvert = False):
    filename = dir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['rho', 'Bcc1', 'Bcc2', 'Bcc3', 'r0'])
    x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']
    rho = data['rho']
    r0 = data['r0']
    b1 = transpose(data['Bcc1'])
    b2 = transpose(data['Bcc2'])
    b3 = transpose(data['Bcc3'])
    
    print("Bx = ",b1.min(), b1.max())
    # ii = input("B")
    
    b = sqrt(b1**2+b2**2+b3**2)
    print("max B",b[isfinite(b)].max())
    # print("max B(z<0)",b[z<1.].max())
    # ii = input("B")
    bmax = b[isfinite(b)].max()
    wfield = b<(bmax*0.1)
    b1[wfield]=sqrt(-1)
    b2[wfield]=sqrt(-1)
    b3[wfield]=sqrt(-1)

    #b = ma.masked_array(b, mask = (b<= bmax/100.))
    #b1 = ma.masked_array(b1, mask = (b<= bmax/100.))
    #b2 = ma.masked_array(b2, mask = (b<= bmax/100.))
    #b3 = ma.masked_array(b3, mask = (b<= bmax/100.))
    x3, y3, z3 = meshgrid(x,y,z, indexing='ij')
    xmax = x.max() - y.min()
    t = data['Time']
    xBH, yBH = BHcoords(t)
    
    if ifradial:
        r = sqrt(x3**2+y3**2+z3**2)
        clf()
        plot(r.flatten(), rho.max()*((r.flatten()/6.)**2/3.+1.)**(-2.5), ',r')
        plot(r.flatten(), rho.flatten(), '.k')
        xscale('log')  ; yscale('log')
        xlabel(r'$R$') ; ylabel(r'$\rho$')
        savefig('radial.png')
    
    allscale = 0.05
    
    fig = maya.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size = (500, 500))
    maya.clf()
    mc = maya.contour3d(x3, y3, z3, transpose(log10(rho)), transparent=True, colormap='jet', opacity=0.2)
    # mc = maya.contour3d(x3, y3, z3, (log10(b+1e-5)), transparent=True, colormap='jet', opacity=0.2)

    fl1 = maya.flow(x3, y3, z3, b1/b, b2/b, b3/b, seedtype='sphere', seed_scale=1.0 * allscale, scalars = log10(b), colormap = 'Oranges', seed_visible=False, seed_resolution=20, vmin = log10(bmax)-3., vmax = log10(bmax), integration_direction = 'both', transparent=True)
    fl2 = maya.flow(x3, y3, z3, b1/b, b2/b, b3/b, seedtype='sphere', seed_scale=5.0 * allscale, scalars = log10(b), colormap = 'Oranges', seed_visible=False, seed_resolution=20, vmin = log10(bmax)-3., vmax = log10(bmax), integration_direction = 'both', transparent=True)

    # direction towards the BH
    rtmp = arange(1000)/1000.
    w = ((rtmp * xBH) >= x.min()) * ((rtmp * xBH) <= x.max()) * ((rtmp * yBH) >= y.min()) * ((rtmp * yBH) <= y.max())
    print(w.sum())
    maya.plot3d(rtmp[w] * xBH, rtmp[w] * yBH, rtmp[w]*0., line_width=0.5, tube_radius=0.25, color=(0,0,0))

    # drawing the original orbit
    nphi = 1000
    phitmp = ((arange(nphi)+0.5)/double(nphi)-0.5) * 2. * pi
    rtmp = 2. * rper / (1.+cos(phitmp))
    xtmp = -rtmp * cos(phitmp) + xBH
    ytmp = -rtmp * sin(phitmp) + yBH
    # print(rtmp[(abs(xtmp-xstar)<xmax)*(abs(ytmp-ystar)<xmax)])
    w = (xtmp >= x.min()) * (xtmp <= x.max()) * (ytmp >= y.min()) * (ytmp <= y.max())
    print(w.sum())
    maya.plot3d(xtmp[w], ytmp[w], rtmp[w]*0., line_width=0.5, tube_radius=0.25, color=(0,1,0))
    # if qname=='r0':
    maya.points3d([0.], [0.], [0.], color=(0,1,0), scale_factor = 1)
    maya.points3d([xBH], [yBH], [0.], color=(0,0,0), scale_factor = 2)

    maya.view(azimuth=30, elevation=70, distance = xmax*2.5, focalpoint = (0., 0., 0.))
    maya.scalarbar(object = mc)
    maya.savefig('magmaya.png')
    
    # vertical extent:
    if ifvert:
        zmin = quantile(z3[r0>0.5], 0.1)
        zmax = quantile(z3[r0>0.5], 0.9)
        zspread = z3[r0>0.5].std()
    
    if ifmovie:
        nm = 300
        az = arange(nm)/double(nm-1) * 360.
        for k in arange(nm):
            maya.view(azimuth=az[k], elevation=70, distance = xmax*3. * (double(nm)-double(k)*0.8)/double(nm), focalpoint = (0., 0., 0.))
            maya.savefig('mmaya{:05d}.png'.format(k))

    if ifvert:
        return t, zmin, zmax, zspread

def hdf3d(nfile, qname = 'rho', dir = 'P1', iflog = False, ifmovie = False):

    filename = dir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = [qname])
    x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']
    q = data[qname]
    if iflog:
        q = log10(q)
    x3, y3, z3 = meshgrid(x,y,z, indexing='ij')

    xmax = x.max() - y.min()

    t = data['Time']
    xBH, yBH = BHcoords(t)

    print("distance to the BH", sqrt(xBH**2+yBH**2))

    fig = maya.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size = (500, 500))
    maya.clf()
    mc = maya.contour3d(x3, y3, z3, transpose(q), transparent=True, colormap='jet', opacity=0.3)
    
    rtmp = arange(1000)/1000.
    w = ((rtmp * xBH) >= x.min()) * ((rtmp * xBH) <= x.max()) * ((rtmp * yBH) >= y.min()) * ((rtmp * yBH) <= y.max())
    print(w.sum())
    maya.plot3d(rtmp[w] * xBH, rtmp[w] * yBH, rtmp[w]*0., line_width=0.5, tube_radius=0.25, color=(0,0,0))

    # drawing the original orbit
    nphi = 1000
    phitmp = ((arange(nphi)+0.5)/double(nphi)-0.5) * 2. * pi
    rtmp = 2. * rper / (1.+cos(phitmp))
    xtmp = -rtmp * cos(phitmp) + xBH
    ytmp = -rtmp * sin(phitmp) + yBH
    # print(rtmp[(abs(xtmp-xstar)<xmax)*(abs(ytmp-ystar)<xmax)])
    w = (xtmp >= x.min()) * (xtmp <= x.max()) * (ytmp >= y.min()) * (ytmp <= y.max())
    print(w.sum())
    maya.plot3d(xtmp[w], ytmp[w], rtmp[w]*0., line_width=0.5, tube_radius=0.25, color=(0,1,0))
    # if qname=='r0':
    maya.points3d([0.], [0.], [0.], color=(0,1,0), scale_factor = 1)
    maya.points3d([xBH], [yBH], [0.], color=(0,0,0), scale_factor = 2)
    # maya.axes()
    maya.view(azimuth=30, elevation=70, distance = xmax*2.5, focalpoint = (0., 0., 0.))
    maya.scalarbar(object = mc)
    maya.savefig('maya.png')
    if ifmovie:
        nm = 100
        az = arange(nm)/double(nm-1) * 360.
        for k in arange(nm):
            maya.view(azimuth=az[k], elevation=70, distance = xmax*2., focalpoint = (0., 0.,0.))
            maya.savefig('maya{:05d}.png'.format(k))

def nhdf3d(n1,n2, dir='M5'):

    zsize = zeros(n2-n1)
    tar = zeros(n2-n1)
    fout = open(dir+'/zsizes.dat', 'w')
    
    for k in arange(n2-n1)+n1:
        t, z1, z2, dz = magplot(k, dir='M5')
        zsize[k] = dz
        tar[k] = t
        print(t, z2-z1, dz)
        # hdf3d(k, iflog=False, qname='rho', dir='P1', ifmovie=False)
        os.system("cp maya.png maya{:05d}.png".format(k))
        fout.write(str(t)+' '+str(dz)+'\n')
        
    fout.flush()
    fout.close()
    
    clf()
    plot(tar, zsize, '-k')
    xlabel(r'$t$')
    ylabel(r'$\Delta z$')
    savefig('zsizes.png')

def spreadcompare():
    
    lines = loadtxt('nspreads.dat') # loading global simulation results
    t_glo = lines[:,0]
    dz_glo = lines[:,-1]

    lines = loadtxt('M5/zsizes.dat') # loading global simulation results
    t_star = lines[:,0]
    dz_star = lines[:,-1]
    
    clf()
    plot(t_glo, dz_glo, 'k:')
    plot(t_star+10., dz_star, 'r-')
    xlabel(r'$t$')  ; ylabel(r'$\Delta z$')
    savefig('spreadcompare.png')

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'maya0*.png' -pix_fmt yuv420p -b 4096k -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" maya.mp4
