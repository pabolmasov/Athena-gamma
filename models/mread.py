from numpy import *
from matplotlib import gridspec

import os
import sys
import glob
sys.path.append("/Users/pasha/athena/vis/python")
import athena_read

from scipy.optimize import root_scalar

ifplot = True
if(ifplot):
    import matplotlib
    from matplotlib.pyplot import *

tper = 10.0
rper = 30.0
addmass = 1.0e6

ecc = 1.0

def time_from_anomaly(f, dt):
    # Barker's equation
    tf = tan(f/2.)
    return sqrt(2.*rper**3/addmass)*tf*(1.+tf**2/3.) - dt
    
def BHcoords(time):
    # finding the root of Barker's equation
    f = root_scalar(time_from_anomaly, args=(time-tper), bracket = [-pi*0.99, pi*0.99])
    print("true anomaly ",f)
    r = 2.*rper/(1.+ecc*cos(f.root))
    x = r * cos(f.root)
    y = r * sin(f.root)
    
    return x,y

def someslices(nar = arange(10), ddir = 'C1'):
    
    nar = nar[::-1]
    nfiles = size(nar)
    nbins = 100
    kc = 0
    
    # clf()
    for k in nar:
        filename = ddir+'/pois.out1.'+'{:05d}'.format(k)+'.athdf'
        data = athena_read.athdf(filename, quantities = ['rho'])
        x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']
        x3, y3, z3 = meshgrid(x, y, z)
        r = sqrt(x3**2+y3**2+z3**2)
        rho = data['rho']

        # radial binning
        if k == nar[0]:
            rbins = arange(nbins+1)/double(nbins) * r.max()
            rhoav = zeros([nbins, nfiles])
            drhoav = zeros([nbins, nfiles])
            rbins_c = (rbins[1:]+rbins[:-1])/2.
            rbins_s = (rbins[1:]-rbins[:-1])/2.
            tar = zeros(nfiles)
            print(rbins)
        for q in arange(nbins):
            w = (r>rbins[q]) & (r<= rbins[q+1])
            rhoav[q, kc] = rho[w].mean()
            drhoav[q, kc] = rho[w].std()
        # a = argsort(r)
        if k == nar[0]:
            tmax = data['Time']
        tar[kc] = data['Time']
        # errorbar(rbins_c, rhoav, yerr = drhoav, xerr = rbins_s, color = (data['Time']/tmax, 0., 0.))
        print("k = ", k)
        kc += 1
    # colorbar()
    # xscale('log') ; yscale('log')
    # savefig('Cslices.png')

    # two-dimensional plots:
    clf()
    pcolormesh(tar, rbins, log10(rhoav))
    colorbar()
    yscale('log')
    xlabel('time') ; ylabel(r'$R/R_\odot$')
    ylim(rbins[1]/3., rbins[-1])
    savefig('Cslices.png')
    clf()
    pcolormesh(tar, rbins, drhoav/rhoav)
    colorbar()
    yscale('log')
    xlabel('time') ; ylabel(r'$R/R_\odot$')
    ylim(rbins[1]/3., rbins[-1])
    savefig('Cslices_d.png')

def hdfmovie():
    n1 = 0 ; n2 = 250
    for k in arange(n2-n1)+n1:
        hdfplot(k, qname='r0', iflog=False)

def readcons(nfile, dir ='windy'):
    filename = dir+'/wc.out4.'+'{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['Etot'])
    x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']

    q = data['Etot']

    print(q.min(), q.max())

    print(shape(q))
    zslice = int(rint(size(z)/2.))

    clf()
    fig = figure()
    pcolormesh(x, y, q[zslice, :,:])
    colorbar()
    savefig('constest.png')

def hdfplot(nfile, ifv = False, ifm = False, qname = 'rho', dir = 'windy', iflog = False):

    filename = dir+'/from_array.prim.'+'{:05d}'.format(nfile)+'.athdf'
    # filename = dir+'/wc.out1.'+'{:05d}'.format(nfile)+'.athdf'
    
    if ifv:
        gad = 5./3.
        data = athena_read.athdf(filename, quantities = ['vel1', 'vel2', 'vel3', 'rho', 'press', 'phi'])
        rho = data['rho'] ; press = data['press']
        vx = data['vel1'] ; vy = data['vel2']  ; vz = data['vel3']
        phi = data['phi']
        x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']
        print("size(x) = ", size(x))
        zslice = int(rint(size(z)/2.))
        z3, y3, x3 = meshgrid(z, y, x, indexing='ij')
        # print(shape(x3))
        # print(shape(vy))
        r = sqrt(x3*x3+y3*y3+z3*z3)
        vphi = ((vy * x3 - vx * y3) / r)[zslice,:,:]
        vr = ((vy * y3 + vx * x3 + vz*z3) / r)[zslice,:,:]
        bernoulli = (vz**2+vy**2+vx**2)/2. + phi + gad/(gad-1.) * press/rho
    else:
        if ifm:
            data = athena_read.athdf(filename, quantities = [qname, 'vel1', 'vel2', 'vel3'])
        else:
            data = athena_read.athdf(filename, quantities = [qname])
        x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']
        zslice = int(rint(size(z)/2.))
        q = data[qname][zslice,:,:]
        print(q.min(), q.max())
        dx = (x.max()-x.min())/double(size(unique(x)))
        dy = (y.max()-y.min())/double(size(unique(y)))
        dz = (z.max()-z.min())/double(size(unique(z)))
        mass = (data[qname]).sum() * dx * dy * dz
        print("total mass = ", mass)
    if ifv:
        clf()
        pcolormesh(x, y, vphi) # , vmin = vphi[r>10.].min(), vmax = vphi[r>10.].max())
        xlabel(r'X') ; ylabel(r'Y')
        colorbar()
        nphi = 1000 ; rper = 50. ; rzero = 200.
        phi0 = arccos(1.-2. * rper/rzero)
        phitmp = (arange(nphi)+0.5)/double(nphi) * 2.*pi
        rtmp = 2. * rper / (1.-cos(phitmp+phi0))
        plot(rtmp * cos(phitmp), rtmp * sin(phitmp), 'w:')
        contour(x,y, vphi, levels=[0.14], colors = 'w')
        xlim(x.min(), x.max())
        ylim(y.min(), y.max())
        title(r't = '+str(data['Time']))

        savefig('Vphi'+'{:05d}'.format(nfile)+'.png')
        clf()
        pcolormesh(x, y, vr)
        xlabel(r'X') ; ylabel(r'Y')
        colorbar()
        plot(rtmp * cos(phitmp), rtmp * sin(phitmp), 'w:')
        xlim(x.min(), x.max())
        ylim(y.min(), y.max())
        title(r't = '+str(data['Time']))

        savefig('Vr'+'{:05d}'.format(nfile)+'.png')
        
        # vxnorm = vx / sqrt(vx**2 + vy**2)
        # vynorm = vy / sqrt(vx**2 + vy**2)
        # rho = data['rho']
        # press = data['press']
        
        alias = 1
        cs = sqrt(gad * press / rho)[zslice, ::alias, ::alias]

        s = log10(data['press']/data['rho']**gad)[zslice, :, :]

        v = squeeze(sqrt(vx**2 + vy**2 + vz**2)[zslice, ::alias, ::alias])

        x3, y3, z3 = meshgrid(z,y,x, indexing='ij')

        vr = (vx * x3 + vy * y3 + vz * z3) / sqrt(x3**2+y3**2+z3**2)

        x2, y2 = meshgrid(x,y)

        clf()
        fig, ax = subplots(1,1)
        pc = pcolormesh(x, y, log10(rho[zslice,:,:]), vmin = -5, vmax = 0. )
        c1=colorbar()
        c1.set_label(r'$\log_{10}\rho$')
        # quiver(x[::alias], y[::alias], (vx)[zslice, ::alias, ::alias]/sqrt(v), (vy)[zslice, ::alias, ::alias]/sqrt(v), log10(v/cs), cmap='Greys', width =0.01)
        # v = ma.masked_array(v, mask = (rho[zslice, ::alias, ::alias] < (1e-4*rho.max())))
        
        c2 = None
        if (v.max() > (cs.max()*0.01)):
            streamplot(x[::alias], y[::alias], squeeze(vx[zslice, ::alias, ::alias]), squeeze(vy[zslice, ::alias, ::alias]), color = log10(v/cs), cmap='Greys')
            c2=colorbar()
            c2.set_label(r'$\log_{10}v/c_{\rm s}$')
        print("Mach = ", (v/cs).min(), median(v/cs), (v/cs).max())
        # xlim(-2.,2.)
        #  ylim(-2., 2.)
        rtmp = arange(500)/500.
        xBH, yBH = BHcoords(data['Time'])
        w = ((rtmp * xBH) >= x.min()) * ((rtmp * xBH) <= x.max()) * ((rtmp * yBH) >= y.min()) * ((rtmp * yBH) <= y.max())
        plot(rtmp[w] * xBH, rtmp[w] * yBH, 'k:')
        title(r't = '+str(data['Time']))
        xlabel(r'$X$') ; ylabel(r'$X$')
        fig.set_size_inches(10.,6.)
        axis('equal')
        savefig('Q'+'{:05d}'.format(nfile)+'.png')
        clf()
        fig, ax = subplots(1,1)
        pc = pcolormesh(x, y, bernoulli[zslice,:,:])
        c1=colorbar()
        c1.set_label(r'$B$')
        print("Bernoulli integral = ", bernoulli.min(), bernoulli.max())
        xlim(-5.,5.)
        ylim(-5., 5.)
        title(r't = '+str(data['Time']))
        # fig.set_size_inches(8.+1.*(c2 is not None),6.)
        ax.axis('equal')
        savefig('B'+'{:05d}'.format(nfile)+'.png')
        close()
    else:
        clf()
        fig = figure()
        if iflog:
            pcolormesh(x, y, log10(q))
        else:
            pcolormesh(x, y, q)
        
        # star surface:
        phitmp = 2.*pi*arange(1000)/double(999)
        rstar = 1.
        plot(rstar * cos(phitmp), rstar * sin(phitmp), 'w')
        xlabel(r'X') ; ylabel(r'Y')
        cs = colorbar()
        cs.set_label(r'$\log_{10}\rho$')
        '''
        nphi = 1000 ; rper = 50. ; rzero = 200.
        phi0 = arccos(1.-2. * rper/rzero)
        phitmp = (arange(nphi)+0.5)/double(nphi) * 2.*pi
        rtmp = 2. * rper / (1.-cos(phitmp+phi0))
        plot(rtmp * cos(phitmp), rtmp * sin(phitmp), 'w:')
        '''
        plot([x.min(), x.min()], [y.min(), y.max()], 'g:')
        plot([x.max(), x.max()], [y.min(), y.max()], 'g:')
        plot([x.min(), x.max()], [y.min(), y.min()], 'g:')
        plot([x.min(), x.max()], [y.max(), y.max()], 'g:')
        xlim(x.min(), x.max())
        ylim(y.min(), y.max())
        title(r't = '+str(round(data['Time'])))
        fig.set_size_inches(8.,6.)
        savefig('XY'+'{:05d}'.format(nfile)+'.png')
        if ifm:
            r1 = 2. ; r2 = 3.
            z3, y3, x3 = meshgrid(z,y,x, indexing='ij')
            r3 = sqrt(x3**2+y3**2+z3**2)
            
    if (qname == 'r0'):
        print('tracer range ', q.min(), q.max())
    if (qname == 'rho') and (qname == 'press'):
        z3, y3, x3 = meshgrid(z,y,x, indexing='ij')
        r3 = sqrt(x3**2+y3**2+z3**2)
        clf()
        plot(r3.flatten(), data[qname].flatten(), '.')
        xscale('log') ; yscale('log')
        xlabel(r'$R$')
        ylabel(r'$\rho$')
        savefig('rhorad.png')
        if ifv:
            rho = data['rho'].flatten()
            press = data['press'].flatten()
            nrho=1000
            rhotmp = (rho.max()/rho.min())**(arange(nrho)/double(nrho-1)) * rho.min()
            clf()
            plot(data['rho'].flatten(), data['press'].flatten(), '.k')
            plot(rhotmp, (rhotmp/rho.max())**(5./3.)*press.max(), '-r')
            xscale('log') ; yscale('log')
            ylabel(r'$P$')
            xlabel(r'$\rho$')
            savefig('EOSplot.png')
            clf()
            plot(r3.flatten(), -data['phi'].flatten(), '.k', label=r'-$\Phi$')
            plot(r3.flatten(), (gad/(gad-1.)*press/rho).flatten(), '.r', label=r'$H$')
            plot(r3.flatten(), (gad/(gad-1.)*press/rho).flatten()+.1, 'xr', label=r'$H$')
            plot(r3.flatten(), (vx**2+vy**2+vz**2).flatten()/2., '.g', label=r'$v^2/2$')
            ylim(-data['phi'].max()*0., -data['phi'].min()*2.)
            legend()
            xscale('log') # ; yscale('log')
            ylabel(r'$B$')
            xlabel(r'$R$')
            savefig('Bplot.png')
    # close('all')
       
        
def nplot(n1, n2):

    # tl =[] ; ml = []

    # fmcurve = open('mcurve.dat', 'w')

    for k in arange(n2-n1)+n1:
        print("n = ", k)
        # tk, mk = hdfplot(k, ifm=True)
        hdfplot(k, ifv=False, dir='windy')
        # tl.append(tk)  ; ml.append(mk)
        # fmcurve.write(str(tk)+' '+str(mk)+'\n')
        
    # fmcurve.flush()
    # fmcurve.close()
        
    # t = asarray(tl) ; m = asarray(ml)
        
def mplot():
    lines = loadtxt('mcurve.dat')
    t = lines[:,0]  ; m = lines[:,1]
    
    tmax = t[m.argmax()]
    mmax = m.max()
        
    rp = 50. ; r0 = 200.
    nu0 = arccos(2.*rp / r0 - 1.)
    t0 = sqrt(2.*rp**3) * tan(nu0/2.) * (1.+tan(nu0/3.)**2/3.)
    print(nu0)
        
    clf()
    fig = figure()
    plot(t, mmax * (t/tmax)**(-4./3.), 'r-')
    plot(t, mmax * (t/tmax)**(-5./3.), 'g-')
    plot(t, m, 'k.')
    plot([t0, t0], [0., m.max()*1.2], 'b:')
    xscale('log') ; yscale('log')
    ylim(0., m.max()*1.2)
    xlabel(r'$t$, $GM_{\rm BH}/c^3$')
    ylabel(r'$\dot{M}$, $M_\odot\, {\rm yr}^{-1}$')
    fig.set_size_inches(12.,4.)
    savefig('mcurve.png')

def sslice():

    # x = R/R*
    nx = 1000
    x = arange(nx)/double(nx)
    
    xc = 0.25 # rcore/rstar

    rho = 1./(1.+(x/xc)**2)

    
