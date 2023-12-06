from numpy import *
# from matplotlib import gridspec

import os
import sys
import glob
from os.path import exists
sys.path.append("../vis/python")
import athena_read

import h5py

from scipy.optimize import root_scalar

# globals:
addmass = 1e3
rper = 30. 
rzero = 600.
f0 = arccos(2.*rper/rzero-1.)
t0 = sqrt(2.*rper**3/addmass)*tan(f0/2.)*(1.+tan(f0/2.)**2/3.)
print("f0 = ", f0)
print("t0 = ", t0)

ifplot = True
if(ifplot):
    import matplotlib
    from matplotlib.pyplot import *

def time_from_anomaly(f, dt):
    tf = tan(f/2.) 
    return sqrt(2.*rper**3/addmass)*tf*(1.+tf**2/3.) - dt 
    
def star_coords(time):
    
    f = root_scalar(time_from_anomaly, args=(time-t0), bracket = [-pi*0.99, pi*0.99])
    print("true anomaly ",f)
    r = 2.*rper/(1.+cos(f.root))
    x = r * cos(f.root+f0)
    y = r * sin(f.root+f0)
    
    return x,y
    
def medplot(nfile, yslice = 0., ddir = 'SG2'):
    
    filename = ddir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'

    data = athena_read.athdf(filename, quantities = ['rho'])
    x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']
    ky = abs(y-yslice).argmin()

    clf()
    fig = figure()
    pcolormesh(x, z, log10(data['rho'][:,ky,:]))
    xlabel(r'X') ; ylabel(r'Z')
    colorbar()
    #  axis('equal')
    ylim(-30.,30.)
    xlim(-150., 200.)
    fig.set_size_inches(10.,3.)
    savefig(dir+'/XZ'+'{:05d}'.format(nfile)+'.png', DPI=500)
    
def history(colno = 1, ddir = 'SG2M'):
    lines = loadtxt(ddir+'/pois.hst')
    t = lines[:,0] ; q = lines[:,colno]
    
    n = arange(size(t))
    
    clf()
    scatter(t, q, c=n)
    xlabel(r'$t$') ; ylabel(r'$\Delta t$')
    yscale('log')
    colorbar()
    savefig('history.png')
    
def multislice(nfile, nslices = 3, alias = 1, zrange = [-100.,100.], qua = 'rho', xrange = None, yrange = None):
    ddir = 'SG2'
    filename = ddir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'
    
    data = athena_read.athdf(filename, quantities = [qua])
    # vx = data['vel1'] ; vy = data['vel2']  ; vz = data['vel3']
    x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']
    q = data[qua]
    if alias > 1:
        x=x[::alias]  ; y=y[::alias] # ; z=z[::alias]
        
    nz = size(z)
    
    zs = (arange(nslices)+0.5)/double(nslices)*(zrange[1]-zrange[0])+zrange[0]
    
    qmax = q.max()
    
    for k in arange(nslices):
        kz = abs(z-zs[k]).argmin()
        qslice = q[kz,::alias,::alias]
        
        clf()
        fig = figure()
        pcolormesh(x, y, log10(qslice), vmin = -14., vmax = log10(qmax))
        colorbar()

        nphi = 1000 
        phi0 = arccos(1.-2. * rper/rzero)
        phitmp = (arange(nphi)+0.5)/double(nphi) * 2.*pi
        rtmp = 2. * rper / (1.-cos(phitmp+phi0))
        plot(rtmp * cos(phitmp), rtmp * sin(phitmp), 'w:')
        plot(rtmp*0.+rzero, rzero * (phitmp-pi), 'w:')
        xlabel(r'X') ; ylabel(r'Y')
        if xrange is None:
            xlim(x.min(), x.max())  
        else:
            xlim(xrange[0], xrange[1])            
        if yrange is None:
            ylim(y.min(), y.max())
        else:
            ylim(yrange[0], yrange[1])
        title(r' z= '+'{0:8.2f}'.format(zs[k])) #str(zs[k]))
        fig.set_size_inches(12.,15.)
        savefig(filename+'_'+str(kz)+'.png', DPI=500)
        close()
        print("slice N{0:10d}".format(kz)+" finished, z = {0:8.2f}".format(zs[k])+': '+filename+'_'+str(kz)+'.png')
        
        
def rawset():
    n1 = 0
    n2 = 270
    for k in arange(n2-n1)+n1:
        if exists('Tsq1/pois.out1.'+'{:05d}'.format(k)+'.athdf'):
            print('Tsq1/pois.out1.'+'{:05d}'.format(k)+'.athdf')
            rawplot(k, qua = 'Bcc3', minlev = 1, maxlev = 5, zslice = 0., dir = 'Tsq1', vrange = [-1.e-5, 1.e-5], starwindow = True, ifcbar = True, starwindowsize=30.+double(k)/2.)
            # rawplot(k, qua = 'rho', minlev = 3, maxlev = 5, zslice = 1., dir = 'T1M', vrange = [-14.,-6.], starwindow = True, ifcbar = True)
        close('all')
        
def cubeset(ddir = 'T3M', narray = [300]):

    dx = 50. 
    dz = 50.
    
    nx = 100
    
    for k in narray:
        if exists(ddir+'/pois.out1.'+'{:05d}'.format(k)+'.athdf'):
            # unicube(k, nx=200, ny=200, nz=100, ddir=ddir, xrange = [-dx, dx], yrange=[-dx, dx], zrange=[-dz, dz], qua='press')
            unicube(k, nx=nx, ny=nx, nz=nx, ddir=ddir, xrange = [-dx, dx], yrange=[-dx, dx], zrange=[-dz, dz], qua='rho')
            unicube(k, nx=nx, ny=nx, nz=nx, ddir=ddir, xrange = [-dx, dx], yrange=[-dx, dx], zrange=[-dz, dz], qua='r0')
            unicube(k, nx=nx, ny=nx, nz=nx, ddir=ddir, xrange = [-dx, dx], yrange=[-dx, dx], zrange=[-dz, dz], qua='Bcc1')
            unicube(k, nx=nx, ny=nx, nz=nx, ddir=ddir, xrange = [-dx, dx], yrange=[-dx, dx], zrange=[-dz, dz], qua='Bcc2')
            unicube(k, nx=nx, ny=nx, nz=nx, ddir=ddir, xrange = [-dx, dx], yrange=[-dx, dx], zrange=[-dz, dz], qua='Bcc3')
            
    
def unicube(nfile, xrange = [-100., 100.], yrange = [-100., 100.], zrange = [-30., 30.], qua = 'rho', nx = 100, ny = 101, nz = 102, ddir = 'SG3', minlev = 1):
    '''
    makes a uniform low-resolution datacube out of an AMR snapshot
    '''
    filename = ddir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = [qua], raw = True)
    
    levels = data['Levels']
    # first coordinate is everywhere the number of the volume    
    nlevels = size(levels)
    
    xstar, ystar = star_coords(data['Time'])

    xout = (xrange[1]-xrange[0])*arange(nx)/double(nx-1)+xrange[0] + xstar
    yout = (yrange[1]-yrange[0])*arange(ny)/double(ny-1)+yrange[0] + ystar
    zout = (zrange[1]-zrange[0])*arange(nz)/double(nz-1)+zrange[0]
    
    qout = zeros([nx,ny,nz])
    
    print("nlevels = ",nlevels)
    
    hfile = h5py.File(filename+'_'+qua+'_tabbox.hdf5', 'w')
    glo = hfile.create_group('globals')
    glo.attrs['Time'] = data['Time']
    glo.attrs['xstar'] = xstar
    glo.attrs['ystar'] = ystar

    geom = hfile.create_group('geometry')
    geom.create_dataset("x", data=xout)
    geom.create_dataset("y", data=yout)
    geom.create_dataset("z", data=zout)
    
    hfile.flush()
    
    for l in arange(nlevels):
        if levels[l] >= minlev:
            x = data['x1v'][l,:] 
            y = data['x2v'][l,:] 
            z = data['x3v'][l,:] 
            # print('box N',l)
            q = data[qua][l,:]
            for i in arange(size(x)):
                if (x[i]>=xout[0]) & (x[i]<xout[-1]):
                    iout = abs(xout-x[i]).argmin()
                    for j in arange(size(y)):
                        if (y[j]>=yout[0]) & (y[j]<yout[-1]):
                            jout = abs(yout-y[j]).argmin()
                            for k in arange(size(z)):
                                if (z[k]>=zout[0]) & (z[k]<zout[-1]):
                                    kout = abs(zout-z[k]).argmin()
                                    #if q[k,j,i]>0.:
                                    #    print(iout, jout, kout)
                                    # if qout[iout, jout, kout] <=0.:
                                    qout[iout, jout, kout] = q[k,j,i]
                                    # if q[k,j,i] > 1e-6:
                                        # print(iout, jout, kout,": ", qout[iout, jout, kout])
    print("writing ",qua)
    q = hfile.create_group(qua)
    q.create_dataset(qua, data=qout)

    hfile.flush()
    hfile.close()
    
def plotbox(nfile, qua = 'rho', nslices = 3, ddir = 'SG3', vrange=[-1e-9, 1e-9], iflog=False):
    
    filename = ddir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'
    hfile = h5py.File(filename+'_'+qua+'_tabbox.hdf5', 'r')
    
    geom=hfile["geometry"]
    glo=hfile["globals"]
    x = geom['x'][:] ; y = geom['y'][:]  ; z = geom['z'][:]
    
    x2, y2 = meshgrid(x,y)

    print(hfile[qua][qua])

    
    q = hfile[qua][qua][:]
    hfile.close()
    
    zmax = z.max() ; zmin = z.min()
    zslices = (zmax - zmin) * arange(nslices)/double(nslices-1)+zmin
    
    print(shape(q))
    print(size(x), size(y), size(z))
    
    clf()
    fig = figure()
    for k in arange(nslices):
        kz = abs(z-zslices[k]).argmin()
        subplot(1, nslices, k+1)
        print(q[:,:, kz].min(), q[:,:, kz].max())
        if iflog:
            pc = pcolormesh(x, y, transpose(log10(q[:,:, kz])), vmin= log10(vrange[0]), vmax=log10(vrange[1]))
        else:
            pc = pcolormesh(x, y, transpose((q[:,:, kz])), vmin= vrange[0], vmax=vrange[1])
        title(r"$z = "+str(round(zslices[k], 3))+"$")
    colorbar()
    fig.set_size_inches(3.*nslices, 3.)    
    savefig('plotbox.png')    
    
    
def dangmo(nfile, minlev = 3, maxlev = 5, dir = 'SG3'):
    '''
    angular momentum distribution
    '''
    filename = dir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'

    data = athena_read.athdf(filename, quantities = ['rho', 'vel1', 'vel2'], raw = True)
    levels = data['Levels']

    nlevels = size(levels)

    njbins = 100
    j0 = sqrt(2. * rper)
    jbins = j0 * 3. * arange(njbins+1)/double(njbins)
    mbins = zeros(njbins)
    
    fout = open(filename+'_jdist.dat', 'w')
    
    for k in arange(nlevels):
        if (levels[k]>=minlev) * (levels[k]<=maxlev):
            x = data['x1v'][k,:] 
            y = data['x2v'][k,:] 
            z = data['x3v'][k,:] 
            dx = data['x1f'][k,1]-data['x1f'][k,0]
            dy = data['x2f'][k,1]-data['x2f'][k,0]
            dz = data['x3f'][k,1]-data['x3f'][k,0]
            dv = dx * dy * dz
            dm = data['rho'][k,:] * dv
            vx = data['vel1'][k,:] ; vy = data['vel2'][k,:]
            j = (x * vy - y * vx) # net angular momentum

            for q in arange(njbins):
                w = (j>=jbins[q]) * (j<jbins[q+1])
                if w.sum() > 0:
                    mbins[q] = (dm*w).sum() # total mass in the j bin
                    # how to estimate the uncertainties?
                fout.write(str(jbins[q])+' '+str(jbins[q+1])+' '+str(mbins[q])+'\n')
    fout.flush()
    fout.close()
    
    clf()
    plot((jbins[1:]+jbins[:-1])/2., mbins, 'k.')
    errorbar((jbins[1:]+jbins[:-1])/2., mbins, fmt='k.', xerr = (jbins[1:]-jbins[:-1])/2.)
    plot(jbins[1:]*0.+j0, mbins, 'r:')
    xlabel(r'$j$') ; ylabel(r'$dM/dj$')
    xscale('log') ; yscale('log')
    savefig(dir+'/jdist.png')
    
def rawplot_x(nfile, maxlev = 10, minlev = 0, qua = 'rho', dir = 'T0M', ifcbar = False, iflog = True, vrange = None, yzrange = [10., 10.]):
    '''
    the same as rawplot, by in the xz plot containing the star
    '''
    filename = dir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'
    if qua == 'b':
        data = athena_read.athdf(filename, quantities = ['Bcc2', 'Bcc3'], raw = True)
    else:
        if qua == 'beta':
            data = athena_read.athdf(filename, quantities = ['Bcc1', 'Bcc2', 'Bcc3', 'press'], raw = True)
        else:
            data = athena_read.athdf(filename, quantities = [qua], raw = True)
    
    levels = data['Levels']
    # first coordinate is everywhere the number of the volume   
    nlevels = size(levels)

    print("maximal level is ", levels.max())

    modifier = '_'+qua
    
    xstar, ystar = star_coords(data['Time'])
    
    dy = yzrange[0] ;  dz = yzrange[1] # ;  dz = 10.
    
    print("XY = ", xstar, ystar)
    
    rlist = [] ; plist = []

    if qua == 'b':
        if vrange is None:
            vmin = (abs(data['Bcc3'][data['Bcc3']>0.])).mean() ; vmax = data['Bcc3'].max()
        else:
            vmin = vrange[0]
            vmax = vrange[1]
        print("b range = ",vmin, vmax)
    else:
        if qua == 'beta':
            # press = data['press']
            # b1 = data['Bcc1'] ; b2 = data['Bcc2'] ; b3 = data['Bcc3']
            vmin = 1.0 ; vmax = 1e5
        else:
            if vrange == None:
                vmin = data[qua].min() ; vmax = data[qua].max()
            else:
                vmin = vrange[0]
                vmax = vrange[1]
            print("range ", vmin, "..", vmax )
    
    # xstar=130. # !!! temporary
    
    clf()
    fig = figure()
    for k in arange(nlevels):
        if (levels[k]>=minlev) * (levels[k]<=maxlev):
            x = data['x1v'][k,:] 
            y = data['x2v'][k,:] 
            z = data['x3v'][k,:] 
            # print(x.min(), x.max())
            if (x.max() >= (xstar-1.))*(x.min() <= (xstar+1.))*(y.max() >= (ystar-dy))*(y.min() <= (ystar+dy))*(z.max() >= (-dz))*(z.min() <= (dz)):
                kxslice = abs(x-xstar).argmin()
                if qua == 'b':
                    qx = data['Bcc1'][k, :, :, kxslice]
                    qy = data['Bcc2'][k, :, :, kxslice]
                    qz = data['Bcc3'][k, :, :, kxslice]
                    print("bx range = ", qx.min(), qx.max())
                    print("by range = ", qy.min(), qy.max())
                    print("bz range = ", qz.min(), qz.max())
                    pc = pcolormesh(y, z, 0.5*log10(qx**2+qy**2+qz**2), vmin = vmin, vmax = vmax)
                    streamplot(y, z, qy, qz, density = 1.0, color='w')
                else:
                    if qua == 'beta':
                        q = (data['press']/(data['Bcc1']**2+data['Bcc2']**2+data['Bcc3']**2))[k, :, :, kxslice] * 2.
                    else:
                        q = data[qua][k, :, :, kxslice]
                    if iflog:
                        pc = pcolormesh(y, z, log10(q), vmin = log10(vmin), vmax = log10(vmax))
                    else:
                        pc = pcolormesh(y, z, q, vmin = vmin, vmax = vmax)
                    if qua == 'beta':
                        qy = data['Bcc2'][k, :, :, kxslice]
                        qz = data['Bcc3'][k, :, :, kxslice]
                        streamplot(y, z, qy, qz, density = 0.5, color='w')
                        contour(y,z, data['press'][k, :, :, kxslice], levels = [1e-9, 1e-7], colors='w')
                        w = isfinite(q)
                        if w.sum() > 0:
                            print("level = ", levels[k],': <beta> = ',q[w].min(),'..',q[w].max())
                        wnan = where(isnan(q))
                        if size(wnan) > 1:
                            rr = (x-xstar)**2 + (y-ystar)**2 + (z-zstar)**2
                            print("radii of NaNs = ", sqrt(rr[wnan].min()), '..', sqrt(rr[wnan].max()))
                    else:
                        print("level = ", levels[k],': <beta> = ',q.min(),'..',q.max())
                plot([y.min(), y.max()], [z.min(), z.min()], 'g:')
                plot([y.min(), y.max()], [z.max(), z.max()], 'g:')
                plot([y.min(), y.min()], [z.min(), z.max()], 'g:')
                plot([y.max(), y.max()], [z.min(), z.max()], 'g:')
    if ifcbar:
        cb = colorbar(mappable=pc)
        cb.set_label(r' '+qua)
    plot([ystar-dy/2., ystar+dy/2.], [0., 0.], ':k')
    plot([ystar, ystar], [-dz/2., dz/2.], ':k')

    xlim(ystar-dy, ystar+dy)
    ylim(-dz, dz)
    title(r't = '+str(round(data['Time'])))
    # show()
    if ifcbar:
        fig.set_size_inches(12.,11. )
    else:
        fig.set_size_inches(11.,11.)

    savefig(dir+'/arawYZ'+'{:05d}'.format(nfile)+modifier+'.png')
    close('all')

def slicetest(nfile, qua='Bcc1', dir = 'T3M'):
    filename = dir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = [qua], raw = True)
    
    levels = data['Levels']
    nlevels=size(levels)

    kzslice = 0 ; kyslice = 0
    
    colorsequence = ['k', 'r', 'b', 'g', 'm']
    
    clf()
    
    for k in arange(nlevels):
        x = data['x1v'][k,:] 
        z = data['x3v'][k,:] 
        y = data['x2v'][k,:] 
        
        if (y.max() < 20.) and (y.min() > -20.) and (z.max() < 20.) and (z.min() > -20.):
            q = data[qua][k,kzslice, kyslice, :]
        # print("x shape = ", shape(x))
        # print("q shape = ", shape(q))
            # scatter(x.flatten(), q.flatten(), s = 0.5, c=colorsequence[k % 5])
            plot(x, q, c=colorsequence[k % 5])
    xlabel(r'X')
    ylabel(r' '+qua)
    xlim(580., 620.)
    savefig('slicetest.png')
    
    
def rawplot(nfile, maxlev = 10, minlev = 0, qua = 'rho', zslice = 1., dir = 'T1H', xrange = None, yrange = None, vrange = None, starwindow = False, starwindowsize = 20., ifcbar = False, ifrhotracer = False, ifmass = False, domainnet = True):
    '''
    plots a 2D slice of a 3D AMR simulation
    
    starwindow overrides all the boxes and shows the 20X20 (or starwindowsizeXstarwindowsize) vicinity of the star
    '''
    
    filename = dir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'
    if qua == 'v':
        data = athena_read.athdf(filename, quantities = ['vel1', 'vel2'], raw = True)
    else:
        if qua == 'b':
            data = athena_read.athdf(filename, quantities = ['rho', 'Bcc1', 'Bcc2'], raw = True)
        else:
            if qua == 's':
                data = athena_read.athdf(filename, quantities = ['rho', 'press', 'r1'], raw = True)
            else:              
                if ifrhotracer or ifmass:
                    data = athena_read.athdf(filename, quantities = [qua, 'r0'], raw = True)
                else:
                    data = athena_read.athdf(filename, quantities = [qua], raw = True)
    # print(data['VariableNames'])
    # ii=input('v')
    levels = data['Levels']
    # first coordinate is everywhere the number of the volume
    
    if ifmass:
        masses = []
    
    nlevels = size(levels)
    
    xmin = None  ;  xmax = None  ;   ymin = None  ;  ymax = None
    
    print("maximal level is ", levels.max())

    modifier = '_'+qua
    
    xstar, ystar = star_coords(data['Time'])
    
    rlist = [] ; plist = [] ; r0list = []
    
    clf()
    fig = figure()
    for k in arange(nlevels):
        if (levels[k]>=minlev) * (levels[k]<=maxlev):
            x = data['x1v'][k,:] 
            y = data['x2v'][k,:] 
            z = data['x3v'][k,:] 
            
            if ifmass and (qua == 'rho'):
                masses.append((data['rho'][k,:]*data['r0'][k,:]).mean() * (x.max()-x.min()) * (y.max()-y.min()) * (z.max()-z.min()))

            if (z.max() >= zslice)*(z.min() < zslice):
                
                kzslice = abs(z-zslice).argmin()
                kzslice = int(floor((zslice-z.min())/(z.max()-z.min()) * double(size(z))))
                print("kzslice = ", kzslice)
                if (qua == 'press') or (qua == 'rho') or (qua == 'r1'):
                    q = data[qua][k,kzslice, :, :]
                    pc=pcolormesh(x, y, log10(q), vmin = vrange[0], vmax = vrange[1])
                    print(q.min(), q.max())
                    eps = sqrt(((q[1:,:]-q[:-1,:])**2/(q[1:,:]+q[:-1,:])**2).max()+((q[:,1:]-q[:,:-1])**2/(q[:,1:]+q[:,:-1])**2).max())
                    print("max eps = ", eps)
                    x2, y2 = meshgrid(x-xstar,y-ystar)
                    r = sqrt(x2*x2+y2*y2)
                    
                    if (x2**2+y2**2).min()<10000.:
                        rlist.append(sqrt(x2**2+y2**2))
                        plist.append(q)
                        if ifrhotracer:
                            r0  = data['r0'][k,kzslice, :, :]
                            r0list.append(r0)
                if (qua == 's'):
                    press = data['press'][k,kzslice, :, :]
                    rho = data['rho'][k,kzslice, :, :]
                    r1 = data['r1'][k,kzslice, :, :]
                    gad = 5./3.
                    # if (levels[k] > 3):
                    print('P/rho^gamma = ',(press/rho**gad).min(), (press/rho**gad).max())
                    print('r1 = ', (r1).min(), (r1).max())
                    pc=pcolormesh(x, y, (press/rho**gad-r1)/r1, vmin = vrange[0], vmax = vrange[1])
                if qua == 'v':
                    vx = data['vel1'][k,kzslice, :, :]
                    vy = data['vel2'][k,kzslice, :, :]
                    
                    # vxstar = vx[abs(x-xstar).argmin(), abs(y-ystar).argmin()]
                    # vystar = vy[abs(x-xstar).argmin(), abs(y-ystar).argmin()]
                    rstar = sqrt(xstar**2+ystar**2)
                    
                    vphistar = sqrt(2.*rper)/rstar
                    vrstar = sqrt(2./rstar-vphistar**2)
                    if data['Time']<t0:
                        vrstar *= -1.
                    vxstar = vrstar * xstar/rstar - vphistar * ystar/rstar
                    vystar = vrstar * ystar/rstar + vphistar * xstar/rstar
                    # print("vstar = ", vrstar, vphistar)
                    x2, y2 = meshgrid(x-xstar,y-ystar)
                    r = sqrt(x2*x2+y2*y2)
                    vr = ((vx-vxstar) * x2 + (vy-vystar) * y2) / r
                    pc = pcolormesh(x, y, vr, vmin = vrange[0], vmax = vrange[1])
                if qua == 'b':
                    Bx = data['Bcc1'][k,kzslice, :, :]
                    By = data['Bcc2'][k,kzslice, :, :]
                    rho = data['rho'][k,kzslice, :, :]
                    x2, y2 = meshgrid(x,y)
                    pc = pcolormesh(x, y, log10(sqrt(Bx**2+By**2)), vmin = vrange[0], vmax = vrange[1])
                    Bma = ma.masked_array((sqrt(Bx**2+By**2)), mask= isnan(Bx+By))
                    print("max ", qua," = ", (sqrt(Bx**2+By**2)).max())
                    contour(x,y, log10(rho), colors='w', levels = [-10., -8., -6., -4.])
                    alias = 1
                    streamplot(x[::alias], y[::alias], Bx[::alias, ::alias], By[::alias, ::alias], color = 'w', density = .25) # , cmap = 'Greys_r')

                if (qua == 'Bcc3') or (qua == 'Bcc1') or (qua == 'Bcc2') or (qua == 'vel1') or (qua == 'vel2') or (qua == 'vel3'):
                    print("vz = ", data[qua][k,:,:,:].min(), '...', data[qua][k,:,:,:].max(), ';   level ', levels[k])
                    
                    v = data[qua][k,kzslice, :, :]
                    print(v.min(), v.max())
                    epsv = sqrt(((v[1:,:]-v[:-1,:])**2/(v[1:,:]+v[:-1,:])**2).max()+((v[:,1:]-v[:,:-1])**2/(v[:,1:]+v[:,:-1])**2).max())
                    print("max eps = ", epsv)
                    if vrange is None:
                        vmin = quantile(data[qua],0.01)  ; vmax = quantile(data[qua], 0.99)
                        pc = pcolormesh(x, y, v, vmin=vmin, vmax=vmax)
                    else:
                        pc = pcolormesh(x, y, v, vmin=vrange[0], vmax=vrange[1])
                if (qua == 'r0') or (qua == 'phi'):
                    q = data[qua][k,kzslice, :, :]
                    # x2, y2 = meshgrid(x,y)
                    pc = pcolormesh(x, y, q, vmin=vrange[0], vmax=vrange[1])
                    # print("max ", qua," = ", q.max())
                # if (qua == 'r0') or (qua == 'phi') or (qua == 'b'):
                if domainnet:
                    plot([x.min(), x.max()], [y.min(), y.min()], 'g:')
                    plot([x.min(), x.max()], [y.max(), y.max()], 'g:')
                    plot([x.min(), x.min()], [y.min(), y.max()], 'g:')
                    plot([x.max(), x.max()], [y.min(), y.max()], 'g:')
            if xmin is None:
                xmin = x.min()
            else:
                xmin = minimum(xmin, x.min())
            if xmax is None:
                xmax = x.max()
            else:
                xmax = maximum(xmax, x.max())
            if ymin is None:
                ymin = y.min()
            else:
                ymin = minimum(ymin, y.min())
            if ymax is None:
                ymax = y.max()
            else:
                ymax = maximum(ymax, y.max())
        
    if ifcbar:
        cb = colorbar(mappable=pc)
        cb.set_label(r'$\log_{10}\rho$')
    nphi = 1000
    phi0 = arccos(1.-2. * rper/rzero)
    phitmp = (arange(nphi)+0.5)/double(nphi) * 2.*pi
    rtmp = 2. * rper / (1.-cos(phitmp+phi0))
    plot(rtmp * cos(phitmp), rtmp * sin(phitmp), 'w:')
    plot(rtmp*0.+rzero, rzero * (phitmp-pi), 'w:')
    ytmp = arange(1000)
    plot(rzero - ytmp * sqrt((rzero-rper)/rper), ytmp, 'w:')
    plot([xstar], [ystar], 'kx')
    #  plot([xstar,xstar+vxstar*100.], [ystar,ystar+vystar*100.], 'k-')
    
    xlabel(r'X')  ; ylabel(r'Y')
   
    if xrange is not None:
        xmin = maximum(xmin, xrange[0])            
        xmax = minimum(xmax, xrange[1])            
    if starwindow:
        xmin = xstar-starwindowsize
        xmax = xstar+starwindowsize

    xlim(xmin, xmax)
    if yrange is not None:
        ymin = maximum(ymin, yrange[0])            
        ymax = minimum(ymax, yrange[1])                    
    if starwindow:
        ymin = ystar-starwindowsize
        ymax = ystar+starwindowsize

    ylim(ymin, ymax)

    title(r't = '+str(round(data['Time']))+', '+'z = '+str(zslice))
    # show()
    if ifcbar:
        fig.set_size_inches(10.,8. * (ymax-ymin)/(xmax-xmin))
        # fig.set_size_inches(7.,6. * (ymax-ymin)/(xmax-xmin))
    else:
        fig.set_size_inches(9.,7.5 * (ymax-ymin)/(xmax-xmin))
    #         fig.set_size_inches(5.5,6. * (ymax-ymin)/(xmax-xmin))

    # title(r'z = '+str(zslice))
    savefig(dir+'/arawXY'+'{:05d}'.format(nfile)+modifier+'.png', DPI=500)
    close('all')
    if (qua == 'press') or (qua == 'rho'):
        clf()
        ctr = 0
        plistmax = 0.
        for rr in rlist:
            plot(rr, plist[ctr], 'k.')
            plistmax = max(plistmax, plist[ctr].max())
            ctr+=1
        rscale = 6.
        rtmp = arange(199)/double(198) * (10.*rscale)
        plot(rtmp, plistmax / (1.+(rtmp/rscale)**2/3.)**3, 'b-')
        xscale('log')   
        yscale('log')
        # xlim(5., 7.) 
        # ylim(1e-16,1e-6)
        xlabel(r'$R$')  ; ylabel(r'$P$')
        savefig('rhorad.png')
        if ifrhotracer:
            print(shape(rlist), shape(plist), shape(r0list))
            clf()
            ctr=0
            for p in plist:
                plot(r0list[ctr], p, 'k,')
                ctr+=1
            xscale('log')   
            yscale('log')
            # xlim(5., 7.) 
            # ylim(1e-16,1e-6)
            xlabel(r'$R_0$')  ; ylabel(r'$p$')
            savefig('trhorad.png')
            
        close('all')
        
    if ifmass:
        # print(masses)
        print("M* = ", asarray(masses).sum())
            
def hdfplot(nfile, ifv = False, ifm = False, ifs = False, dir = 'SG2', alias = 1, xrange = None, yrange = None, xstar = 0., ystar = 0.):
    
    filename = dir+'/pois.out1.'+'{:05d}'.format(nfile)+'.athdf'
    
    if ifv:
        data = athena_read.athdf(filename, quantities = ['vel1', 'vel2', 'vel3'], subsample = True, level=1)
        vx = data['vel1'] ; vy = data['vel2']  ; vz = data['vel3']
        x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']
        if alias > 1:
            vx = vx[::alias,::alias, ::alias]
            vy = vy[::alias,::alias, ::alias]
            vz = vz[::alias,::alias, ::alias]
            x=x[::alias]  ; y=y[::alias]  ; z=z[::alias]

        zslice = int(rint(size(z)/2.))
        x2, y2 = meshgrid(x-xstar,y-ystar)
        r = sqrt(x2*x2+y2*y2)
        
        kxstar = abs(x-xstar).argmin() ; kystar = abs(y-ystar).argmin()
        
        vxstar = vx[zslice, kystar, kxstar]  ; vystar = vy[zslice, kystar, kxstar]
        
        vphi = ((vy[zslice,:,:]-vystar) * x2 - (vx[zslice,:,:]-vxstar) * y2) / r
        vr = ((vy[zslice,:,:]-vystar) * y2 + (vx[zslice,:,:]-vxstar) * x2) / r
    else:       
        if ifs:
            data = athena_read.athdf(filename, quantities = ['rho', 'press'], fast_restrict = True, level=1)
        else:
            if ifm:
                data = athena_read.athdf(filename, quantities = ['rho', 'vel1', 'vel2', 'vel3'], fast_restrict = True, level=1)
            else:
                data = athena_read.athdf(filename, quantities = ['rho'], fast_restrict = True, level=1)
            
        x = data['x1v'] ; y = data['x2v']  ; z = data['x3v']
        if alias > 1:
            rho = data['rho'][::alias,::alias, ::alias]
            if ifm:
                vx = data['vel1'][::alias,::alias, ::alias]
                vy = data['vel2'][::alias,::alias, ::alias]
                vz = data['vel3'][::alias,::alias, ::alias]
            x=x[::alias]  ; y=y[::alias]  ; z=z[::alias]
        else:
            rho = data['rho']
            if ifm:
                vx = data['vel1']
                vy = data['vel2']
                vz = data['vel3']
            
        # print(x,y,z)
        # ii = input('xyz')
        zslice = int(rint(size(z)/2.))
        q=rho[zslice,:,:]
        dx = (x.max()-x.min())/double(size(unique(x)))
        dy = (y.max()-y.min())/double(size(unique(y)))
        dz = (z.max()-z.min())/double(size(unique(z)))
        mass = (data['rho']).sum() * dx * dy * dz
        print("total mass = ", mass)   
    if ifv:
        clf()
        fig = figure()
        if (xrange is not None) and (yrange is not None):
            pcolormesh(x, y, vphi, vmin = vphi[(x2>xrange[0])&(x2<xrange[1])&(y2>yrange[0])&(y2<yrange[1])].min(), vmax = vphi[(x2>xrange[0])&(x2<xrange[1])&(y2>yrange[0])&(y2<yrange[1])].max())
            plot([xstar], [ystar], 'xr')
        else:
            pcolormesh(x, y, vphi, vmin = vphi[r>10.].min(), vmax = vphi[r>10.].max())

        xlabel(r'X') ; ylabel(r'Y')
        colorbar()
        contour(x,y, vphi * sqrt(r), levels=[1.0], colors = 'w')
        # xlim(-200.,100.) ; ylim(-100.,100.)
        if xrange is None:
            xlim(x.min(), x.max())  
        else:
            xlim(xrange[0], xrange[1])            
        if yrange is None:
            ylim(y.min(), y.max())
        else:
            ylim(yrange[0], yrange[1])

        fig.set_size_inches(12.,15.)
        savefig(dir+'/Vphi'+'{:05d}'.format(nfile)+'.png', DPI=500)
        clf()
        fig = figure()
        if (xrange is not None) and (yrange is not None):
            pcolormesh(x, y, vr, vmin = vphi[(x2>xrange[0])&(x2<xrange[1])&(y2>yrange[0])&(y2<yrange[1])].min(), vmax = vphi[(x2>xrange[0])&(x2<xrange[1])&(y2>yrange[0])&(y2<yrange[1])&(y2<yrange[1])].max())
        else:
            pcolormesh(x, y, vr)

        xlabel(r'X') ; ylabel(r'Y')
        if xrange is None:
            xlim(x.min(), x.max())  
        else:
            xlim(xrange[0], xrange[1])            
        if yrange is None:
            ylim(y.min(), y.max())
        else:
            ylim(yrange[0], yrange[1])

        # xlim(-200.,100.) ; ylim(-100.,100.)
        colorbar()
        fig.set_size_inches(12.,15.)
        savefig(dir+'/Vr'+'{:05d}'.format(nfile)+'.png', DPI=500)
    else:
        clf()
        fig = figure()
        pcolormesh(x, y, log10(q))
        colorbar()
        # xlim(-200.,100.) ; ylim(-100.,100.)
        nphi = 1000 
        phi0 = arccos(1.-2. * rper/rzero)
        phitmp = (arange(nphi)+0.5)/double(nphi) * 2.*pi
        rtmp = 2. * rper / (1.-cos(phitmp+phi0))
        plot(rtmp * cos(phitmp), rtmp * sin(phitmp), 'w:')
        plot(rtmp*0.+rzero, rzero * (phitmp-pi), 'w:')
        if xrange is None:
            xlim(x.min(), x.max())  
        else:
            xlim(xrange[0], xrange[1])            
        if yrange is None:
            ylim(y.min(), y.max())
        else:
            ylim(yrange[0], yrange[1])
        xlabel(r'X')  ; ylabel(r'X')
        title(r't = '+str(round(data['Time'])))
        fig.set_size_inches(12.,15.)
        savefig(dir+'/XY'+'{:05d}'.format(nfile)+'.png', DPI=500)
    close('all')
    if ifm:
        # mdot estimate 
        r1 = 3.  ; r2 = 5. 
        # the mass accretion rate is averaged in the layer between r1 and r2
        nx = size(x) ; ny = size(y) ; nz = size(z)
        z3, y3, x3 = meshgrid(z,y,x, indexing='ij')
        r3sq = x3**2+y3**2+z3**2
        r3 = sqrt(r3sq)
        vr = (vx * x3 + vy * y3 + vz * z3) / r3
        massflow = -rho * vr * r3sq
        wrange = where((r3 > r1) & (r3 < r2))
        print(size(wrange)," points on the sphere")
        mdot = massflow[wrange].mean() * 4. * pi
        
        mdot *= 6.40627e12 # conversion to Msun/yr
        
        t = data['Time']
        
        # write a small ascii output:
        mout = open(filename+'_m.dat','w')
        mout.write(str(t)+' '+str(mdot)+'\n')
        mout.close()
        
        return t, mdot 
       
def mplot(dir = 'SGTDE1'):
    
    lines = loadtxt(dir+'/mcurve.dat')
    t = lines[:,0]  ; m = lines[:,1]
    
    tmax = t[m.argmax()]
    mmax = m.max()
        
    rp = 100. ; r0 = 600.
    nu0 = arccos(2.*rp / r0 - 1.)
    t0 = sqrt(2.*rp**3) * tan(nu0/2.) * (1.+tan(nu0/2.)**2/3.)
    print(nu0)
        
    clf()
    fig = figure()
    plot(t, mmax * (t/tmax)**(-5./3.), 'r-')
    plot(t, m, 'k.')
    plot(t, m, 'g.')
    plot([t0, t0], [0., m.max()*1.2], 'b:')
    # xscale('log') ; yscale('log')
    ylim(abs(m).min(), m.max()*1.2)
    xlabel(r'$t$, $GM_{\rm BH}/c^3$')
    ylabel(r'$\dot{M}$, $M_\odot\, {\rm yr}^{-1}$')
    fig.set_size_inches(12.,4.)
    savefig(dir+'/mcurve.png')

        
def nplot(n1, n2, ifdats = False, mcurve = True, alias = 3):
    
    dir = 'SG2'
    
    n = arange(n2-n1)+n1
    nn = size(n)
    mdot = zeros(nn)  ; t = zeros(nn)

    fmcurve = open(dir+'/mcurve.dat', 'w')
    
    for k in arange(nn):
        print("n = ", n[k])
        if ifdats:
            filename = dir+'/pois.out1.'+'{:05d}'.format(n[k])+'.athdf'
            # print(filename)
            file_exists = exists(filename+'_m.dat')
            if file_exists:
                if mcurve:
                    lines = loadtxt(filename+'_m.dat')
                    ttmp = lines[0]  ;  mdottmp = lines[1]
                print("file "+filename+"_m.dat exists!")
            else:
                ttmp, mdottmp = hdfplot(n[k], ifm = True, dir=dir, alias=alias)
        else:
            if mcurve:
                ttmp, mdottmp = hdfplot(n[k], ifm = True, dir=dir, alias=alias)
            else:
                hdfplot(n[k], dir=dir)
            # hdfplot(n[k], ifv=True, dir=dir)
        if mcurve:
            t[k] = ttmp ; mdot[k] = mdottmp
            print(ttmp, mdottmp)
            fmcurve.write(str(ttmp)+' '+str(mdottmp)+'\n')

    if mcurve:
        fmcurve.flush()
        fmcurve.close()
        mplot(dir=dir)        

# nplot(0,265)
# cubeset(ddir = 'T3M', narray = 200+arange(150))    
# rawset()
