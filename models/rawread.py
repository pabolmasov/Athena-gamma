from numpy import *
from matplotlib import gridspec

from scipy.integrate import simpson

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

gam = 5./3.

def rawcompare(nfile1, nfile2, y0=0., z0 = 0., dir1 = 'gtest', dir2 = 'gtest_outflow', qua = 'press', iflog=False):
    
    filename1 = dir1+'/pois.out1.'+'{:05d}'.format(nfile1)+'.athdf'
    filename2 = dir2+'/pois.out1.'+'{:05d}'.format(nfile2)+'.athdf'

    data1 = athena_read.athdf(filename1, quantities = [qua], raw = True)
    data2 = athena_read.athdf(filename2, quantities = [qua], raw = True)
    
    # q1 = data1[qua] ; q2 = data2[qua]
    x1 = data1['x1v'] ; x2 = data2['x1v']
    y1 = data1['x2v'] ; y2 = data2['x2v']
    z1 = data1['x3v'] ; z2 = data2['x3v']

    yztol1 = max([abs(y1-y0).min(),abs(z1-y0).min()])
    yztol2 = max([abs(y2-y0).min(),abs(z2-y0).min()])

    print("yztol = ", yztol2)

    levels1 = data1['Levels']
    levels2 = data2['Levels']
    nlevels1 = size(levels1)
    nlevels2 = size(levels2)

    clf()

    fig = figure()

    for k in arange(nlevels1):
        x1 = data1['x1v'][k]
        y1 = data1['x2v'][k]
        z1 = data1['x3v'][k]

        yfilter1 = ((abs(y1-y0) <= yztol1)).argmin()
        zfilter1 = ((abs(z1-z0) <= yztol1)).argmin()
        
        print(yfilter1)
        
        q1 = squeeze(data1[qua][k])
    
        if (yfilter1 >= 0) and (zfilter1 >= 0):
            plot(x1, squeeze(q1[zfilter1, yfilter1, :]), 'k-')
            # print("level "+levels1[k])

    for k in arange(nlevels2):
        x2 = data2['x1v'][k]
        y2 = data2['x2v'][k]
        z2 = data2['x3v'][k]

        yfilter2 = (abs(y2-y0) <= yztol2).argmin()
        zfilter2 = (abs(z2-z0) <= yztol2).argmin()

        q2 = squeeze(data2[qua][k])

        if (yfilter2 >= 0) and (zfilter2 >= 0):
            plot(x2, squeeze(q2[zfilter2, yfilter2,:]), 'r:')
            # print("level "+levels2[k])

    if iflog:
        yscale('log')
    xlabel(r'$X$')
    ylabel(r' '+qua)
    title(r't = '+str(round(data1['Time'], 2))+'; '+str(round(data2['Time'], 2)))
    fig.tight_layout()
    savefig('slicecompare.png')

def rawplot(nfile, maxlev = 10, minlev = 0, qua = 'rho', dir = 'qtest', ifcbar=True, vrange=[0.,1.], ifslice = False, iflog = True):
    '''
    '''
    
    filename = dir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'

    if (qua == 'b'):
        data = athena_read.athdf(filename, quantities = ['Bcc1', 'Bcc2'], raw = True)
    else:
        if qua == 's':
            data = athena_read.athdf(filename, quantities = ['press', 'rho'], raw = True)
        else:
            data = athena_read.athdf(filename, quantities = [qua], raw = True)
    # print(data['VariableNames'])
    # ii=input('v')
    levels = data['Levels']
    # first coordinate is everywhere the number of the volume
    
    nlevels = size(levels)
        
    print("levels ", levels)

    modifier = '_'+qua
        
    xmin = -50. ; ymin = -50.
    xmax = 150. ; ymax = 50.

    zslice = 0.01
    
    clf()
    fig = figure()
    for k in arange(nlevels):
        if (levels[k]>=minlev) * (levels[k]<=maxlev):
            x = data['x1v'][k]
            y = data['x2v'][k]
            z = data['x3v'][k]
            # print("z = ", z.min(), z.max())
            if (z.min()<=zslice) * (z.max()>zslice):
                # print("level = ", levels[k])
                kzslice = (z-zslice).argmin()
                if (qua == 'b'):
                    b1 = data['Bcc1'][k,kzslice,:] ; b2 = data['Bcc2'][k,kzslice,:]
                    q = sqrt(b1**2+b2**2)
                else:
                    if qua == 's':
                        q = data['press'][k,0,:] / data['rho'][k,0,:]**gam
                    else:
                        q = data[qua][k,0,:]
                print(shape(x), shape(q))
                if ifslice:
                    plot(x, q[0,:], 'k-')
                    plot(x, q[32, :], 'r-')
                else:
                    if iflog:
                        pc=pcolormesh(x, y, log10(q), vmin = log10(vrange[0]), vmax=log10(vrange[1]))
                    else:
                        pc=pcolormesh(x, y, q, vmin = vrange[0], vmax=vrange[1])
                if (qua == 'b') and not(ifslice):
                    streamplot(x,y,b1, b2, color='k', density=0.5)
                if not(ifslice):
                    print(q.min(), q.max())
                    xmax = maximum(xmax, x.max())
                    ymax = maximum(ymax, y.max())
                    xmin = minimum(xmin, x.min())
                    ymin = minimum(ymin, y.min())
                    plot([x.min(), x.min()], [y.min(), y.max()], 'g:')
                    plot([x.max(), x.max()], [y.min(), y.max()], 'g:')
                    plot([x.min(), x.max()], [y.min(), y.min()], 'g:')
                    plot([x.min(), x.max()], [y.max(), y.max()], 'g:')
    if ifcbar and not(ifslice):
        cb = colorbar(mappable=pc)
        cb.set_label(r'$\log_{10}P$')
   
    title(r't = '+str(round(data['Time'], 2)))


    if ifcbar:
        fig.set_size_inches(9.,12. * (ymax-ymin)/(xmax-xmin))
        # fig.set_size_inches(7.,6. * (ymax-ymin)/(xmax-xmin))
    else:
        fig.set_size_inches(10.,12. * (ymax-ymin)/(xmax-xmin))
    #         fig.set_size_inches(5.5,6. * (ymax-ymin)/(xmax-xmin))

    # title(r'z = '+str(zslice))
    if ifslice:
        xlabel(r'X')  ; ylabel(r' '+qua)

        savefig('slice.png')
        
    else:
        xlabel(r'X')  ; ylabel(r'Y')
        xlim(xmin, xmax)
        ylim(ymin, ymax)
        fig.tight_layout()
        savefig(dir+'/arawXY'+'{:05d}'.format(nfile)+modifier+'.png')
    close('all')

def rawset():

    n1 = 0
    n2 = 101

    for k in arange(n2-n1)+n1:
        rawplot(k, qua='press', vrange=[1e-15, 1e-7], iflog=True, dir='gtest_multipole')
    close('all')
        
def isostar(r):
    # approximate isothermal solution from from Raga et al (2013)
    if size(r) > 1:
        rho = copy(r)
        for k in arange(size(r)):
            rho[k]=isostar(r[k])
        return rho
    else:
        r0 = 2.312
        c = 0.548
        rsq = r**2
        if (r < r0):
            # near field
            return (1.+(2.*c-1.) * rsq)/(1.+c*rsq)**2
        else:
            q = 0.735 / sqrt(r) * (1.+5.08 * r ** (-1.94)) * cos(sqrt(7.)/2. * log(r) + 5.396 * (1. + 0.92 * r ** (-2.31)))
            return (1.+q)/3./rsq;
        
rmin = 0.001 ; rmax = 1000. ; nr = 10000
r = (rmax/rmin) ** (arange(nr)/double(nr-1)) * rmin

y = isostar(r)

clf()
plot(r, y, 'k-')
xscale('log') ; yscale('log')
savefig('isostar.png')

w = (r < 2.312)

mass = simpson((y*4.*pi*r**2)[w], x=r[w])

print("mass multiplier = ", mass/4./pi)

# making the movie:
# ffmpeg -f image2 -r 15 -pattern_type glob -i 'qtest_noMR/arawXY*_press.png' -pix_fmt yuv420p -b 4096k -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" XY.mp4
