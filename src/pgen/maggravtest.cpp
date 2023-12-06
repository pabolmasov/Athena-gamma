//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file binary_gravity.cpp
//  \brief Problem generator to test Multigrid Poisson's solver with Multipole Expansion

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstring>    // memset
#include <ctime>
#include <iomanip>
#include <iostream>
// #include <cstdlib>
#include <list>
#include <fstream> // reading and writing files
using std::ifstream;
#include <limits>

// #include <gsl/gsl_sf.h> // GNU scientific library
// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/mg_gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if SELF_GRAVITY_ENABLED != 2
#error "This problem generator requires Multigrid gravity solver."
#endif


namespace{
  Real four_pi_G, gam;
  Real rstar, mstar, bgdrho, temp, vstarx, vstary, vstarz;
  Real rcore, amp; // xcore = rstarcore/rstar
  Real denpro(Real x); // density profile inside the star, normalised by unity; argument is the distance from the centre/ R*
  Real ppro(Real x); // pressure profile (consistent with denpro)
  // ifstream indata;
  std::string starmodel; // the name of the MESA file containing the density and pressure profiles
  void instar_interpolate(std::list<Real> instar_radius, std::list<Real> instar_lrho, std::list<Real> instar_lpress, std::list<Real> instar_mass, Real r2, Real rres, Real *rho, Real *p);
  // Real instar_interpolate(std::list<Real> instar_radius, std::list<Real>instar_array, std::list<Real> instar_mass, Real r2, Real rres);
  void psi_AB(Real r, Real cth, Real phi, Real *vr, Real *vth, Real *vphi);
  int rmin, rmax, magbeta; // multipolar structure in magnetic field or convection velocity field
  // Real qx, qy, qz; // perturbation wave vector
}

Real refden, refB, threshold;

int RefinementCondition(MeshBlock *pmb);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  gam = pin->GetReal("hydro","gamma");
  four_pi_G = pin->GetReal("problem","four_pi_G");
  starmodel = pin->GetString("problem", "starmodel");

    // gm0 = pin->GetOrAddReal("problem","GM",0.0);
    rstar = pin->GetReal("problem","rstar");
    rmin = pin->GetReal("problem","rmin");
    rmax = pin->GetReal("problem","rmax");

    magbeta = pin->GetReal("problem","magbeta");

    //    xcore = rstarcore / rstar;
    mstar = pin->GetReal("problem","mstar");
    // fcore = pin->GetReal("problem","fcore");

    bgdrho = pin->GetReal("problem","bgdrho");
    refden = pin->GetReal("problem","refden");
    refB = pin->GetReal("problem","refB");

    threshold = pin->GetReal("problem","thresh");

    // bgdrho = 1e-14; // !!!temporary
    temp = pin->GetReal("problem","temp");
    amp = pin->GetReal("problem","amp");

    vstarx = pin->GetReal("problem","vstarx");
    vstary = pin->GetReal("problem","vstary");
    vstarz = pin->GetReal("problem","vstarz");
    
    SetFourPiG(four_pi_G);

  // AMR setup:
  if (adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for gravity from a binary
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
 //   Real x0 = 12.0/1024.0, y0 = 0.0, z0 = 0.0, r = 6.0/1024.0;
 
  Real bgdp = temp * bgdrho;
  Real G = four_pi_G / (4.0 * PI);
  
  Real dencurrent = 0., pcurrent = 0.;
  
  std::int64_t iseed = -1 - gid;
  
  // density distribution in the star:
  std::list<Real> instar_mass = {};
  std::list<Real> instar_radius = {};
  std::list<Real> instar_lrho = {};
  std::list<Real> instar_lpress = {};
  
  Real tmp_m, tmp_r, tmp_lrho, tmp_lpress, instar_rmax = 0., instar_mtot, mcheck = 0.;
  ifstream indata;
  indata.open(starmodel); // opens the file
  //  std::cout << "reading data from " << starmodel << std::endl ;
  // std::cout << "refB = " << refB << std::endl;
  // std::string s;
  // indata >> s; // header (skipping one line)
  while ( !indata.eof() ) { // keep reading until end-of-file
    indata >> tmp_m >> tmp_r >> tmp_lrho >> tmp_lpress; // sets EOF flag if no value found; one additional column in the model
    std::cout << tmp_m << " "<< tmp_r << " " << tmp_lrho << " " << tmp_lpress << std::endl ;
    instar_mass.push_back(tmp_m);
    instar_radius.push_back(tmp_r);
    instar_lrho.push_back(tmp_lrho);
    instar_lpress.push_back(tmp_lpress/G);
    if (tmp_r > instar_rmax) instar_rmax = tmp_r;
    if (tmp_m > instar_mtot) instar_mtot = tmp_m;
  }
  indata.close();
  
  //   std::cout << "lists read; size = " << instar_mass.size() << " " << instar_radius.size() << " "<< instar_lrho.size() << std::endl ;  
    //    instar_lpress = instar_calculate_pressure(instar_mass, instar_radius, instar_lrho);

  // std::cout << "instar_rmax = " << instar_rmax << std::endl ;
    // std::cout << "r_virial = " << rvir << std::endl ;

  Real starmass_empirical = 0.;

  Real rnorm = (rmax+rmin)/2.;
  // interpolation for velocity normalization
  instar_interpolate(instar_radius, instar_lrho, instar_lpress, instar_mass, rnorm/rstar, 0., &dencurrent, &pcurrent) ;
  std::cout << "d, P = " << dencurrent << ", " << pcurrent << std::endl ;
  Real rcsnorm = std::sqrt(pcurrent) *  SQR((rmin+rmax)/(rmax-rmin)) / 4.; // psi0/2/pi
  if (MAGNETIC_FIELDS_ENABLED) rcsnorm = std::sqrt(2. * pcurrent / magbeta) * (mstar / rstar); // maximal magnetic field is normalized by maximal 
  std::cout << "norm = " << rcsnorm << std::endl;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
          Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k);
          Real r2 = std::sqrt(SQR(x)+SQR(y)+SQR(z));
          Real cth2 = z / r2, phi2 = std::atan2(y,x);
          Real rhogas = bgdrho;
          rhogas = bgdrho;
	  Real dr = sqrt(SQR(pcoord->dx1f(i))+SQR(pcoord->dx2f(j))+SQR(pcoord->dx3f(k))), dv = std::fabs(pcoord->dx1f(i) * pcoord->dx2f(j) * pcoord->dx3f(k));
	  // std::cout << "dx = " << pcoord->dx1f(i) << std::endl;
	  // std::cout << "dy = " << pcoord->dx2f(j) << std::endl;
	  // std::cout << "dz = " << pcoord->dx3f(k) << std::endl;

	  // getchar();
	  // dr = 0. ; // !!! temporary
	  Real vr = 0.,vth = 0., vphi = 0., cs = 0.;
	  if (r2 <= rstar){ // inside the star
	    instar_interpolate(instar_radius, instar_lrho, instar_lpress, instar_mass, r2 / rstar, dr / rstar,
			      &dencurrent, &pcurrent) ; // * pcoord->dx1f(i));
	    cs = std::sqrt(pcurrent / (dencurrent+rhogas)) ; 
	    if((r2>rmin)&&(r2<rmax)){
	      psi_AB(r2, cth2, phi2, &vr, &vth, &vphi); // poloidal loops defined here
	      vr *= rcsnorm ; vth *= rcsnorm ; vphi *= rcsnorm ; // normalization ; if MF are defined, this will be field normalized by magbeta = pgas/pmag; otherwise, convective velocities normalized by the speed of  sound
	    }
	    else{
	      vr = vth = vphi = 0.;
	    }
	    // psi_AB(r2, cth2, phi2, instar_rmax, cs * dencurrent, &vr, &vth, &vphi);
	    //	    vr *= amp; vth *= amp; vphi *= amp;
	    starmass_empirical += dencurrent * dv ;
	    // std::cout << " starmass_cumulative = " << starmass_empirical << " += " << dv << " * " << dencurrent << std::endl; 
	    if(NSCALARS>0)pscalars->s(0,k,j,i) = 1.;
      	}
        else{	
          dencurrent = 0.;
          pcurrent = 0.;
	  if(NSCALARS>0)pscalars->s(0,k,j,i) = 0.;
        }

	  Real vxr = ((vr * std::sqrt(1.-cth2*cth2) + vth * cth2) * cos(phi2) - vphi * sin(phi2)) ; // MF or velocity converted to Cartesian frame
	  Real vyr = ((vr * std::sqrt(1.-cth2*cth2) + vth * cth2) * sin(phi2) + vphi * cos(phi2)) ;
	  Real vzr = (vr * cth2 - vth * std::sqrt(1.-cth2*cth2)) ;

	  // instar_mtot is in mesa units (Msun), and we need to downscale it to the units where MBH = 1
	  // Real randwave = std::cos(qx * x + qy * y + qz * z); // perturbation in the form of a flat wave
	  phydro->w(IDN,k,j,i) = dencurrent * mstar / std::pow(rstar, 3.) + rhogas;
	  // pressure scales with M^2, hence we need to downscale it twice
	  phydro->w(IPR,k,j,i) = pcurrent * SQR(mstar / rstar) + rhogas*temp;

          // phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i);
	// velocities:
	  phydro->w(IM1,k,j,i) = 0.;
	  phydro->w(IM2,k,j,i) = 0.;
	  phydro->w(IM3,k,j,i) = 0.;

      if (r2 < rstar) {
	    if (MAGNETIC_FIELDS_ENABLED) {
	      pfield->b.x1f(k,j,i) = vxr;
	      pfield->b.x2f(k,j,i) = vyr;
	      pfield->b.x3f(k,j,i) = vzr;
	    }
	    else{
	      phydro->w(IM1,k,j,i) = vxr  / std::max(dencurrent,rhogas);
	      phydro->w(IM2,k,j,i) = vyr  / std::max(dencurrent,rhogas);
	      phydro->w(IM3,k,j,i) = vzr  / std::max(dencurrent,rhogas);
	    }
          phydro->w(IM1,k,j,i) = vstarx;
          phydro->w(IM2,k,j,i) = vstary;
          phydro->w(IM3,k,j,i) = vstarz;

      }
	  else{// outside the star
	    if (MAGNETIC_FIELDS_ENABLED) {
              pfield->b.x1f(k,j,i) = 0.;
              pfield->b.x2f(k,j,i) = 0.;
              pfield->b.x3f(k,j,i) = 0.;
            }
	  }

      }
    }
  }

  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is, ie, js, je, ks, ke); //

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie, js, je, ks, ke);

}

int RefinementCondition(MeshBlock *pmb)
{
  // from src/pgen/dmr.cpp

  //  Real den2 = mstar*1. / (4.0*PI * rstar*rstar*rstar) ; // mean star density

  Real dfloor = refden ; // density floor for refinement
  Real bfloor = refB; 
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> &b = pmb->pfield->bcc;

  Real maxeps = 0.0, maxden = 0.0, maxvvar = 0.0, maxbb = 0.0;
  Real rad = 0., vvar = 0., vv, v;

  Real rBH = 30.; // central region, remaining unresolved

  Real refdelta = threshold; // temperary
  
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {
        Real x = pmb->pcoord->x1v(i), y = pmb->pcoord->x2v(j), z = pmb->pcoord->x3v(k);
        Real r1 = std::sqrt(x*x+y*y+z*z); // distance to the BH                                                                                                                                
	Real eps = 0., den = 0., bb=0.,  bsq = 0.;
 
	bool magfilter = true;
	if (MAGNETIC_FIELDS_ENABLED) {
	  bsq = SQR(b(IB1, k, j, i))+SQR(b(IB2, k, j, i))+SQR(b(IB3, k, j, i));
	  magfilter = (bsq > SQR(refB));
	}

	//	Real eps = 0., den = 0., bb=0.,  bsq = 0.;
	if((w(IDN,k,j,i) >= dfloor) && (r1>rBH) && magfilter){
	  Real epsr= (std::abs(w(IDN,k,j,i+1)-2.0*w(IDN,k,j,i)+w(IDN,k,j,i-1))
		       +std::abs(w(IDN,k,j+1,i)-2.0*w(IDN,k,j,i)+w(IDN,k,j-1,i))
		      +std::abs(w(IDN,k+1,j,i)-2.0*w(IDN,k,j,i)+w(IDN,k-1,j,i)))/(w(IDN,k,j,i)+refden);
	  Real epsp= (std::abs(w(IEN,k,j,i+1)-2.0*w(IEN,k,j,i)+w(IEN,k,j,i-1))
		      +std::abs(w(IEN,k,j+1,i)-2.0*w(IEN,k,j,i)+w(IEN,k,j-1,i))
		      +std::abs(w(IEN,k+1,j,i)-2.0*w(IEN,k,j,i)+w(IEN,k-1,j,i)))/(w(IEN,k,j,i)+refden*temp);
	 
	  // maximal tangential velocity shift
	  eps = std::max(epsr, epsp);
	  // eps = epsr;
	  den = w(IDN,k,j,i);
	  // vvar = vv / (v + w(IEN, k, j, i)/w(IDN, k, j, i));
	}
	//	Real x = pmb->pcoord->x1v(i), y = pmb->pcoord->x2v(j), z = pmb->pcoord->x3v(k);
	//	Real r1 = sqrt(x*x+y*y+z*z); // distance to the BH                    

	maxeps = std::max(maxeps, eps);
	maxden = std::max(maxden, den);
	// maxvvar= std::max(maxvvar, vvar);
	if (MAGNETIC_FIELDS_ENABLED) {
	  maxbb = std::max(maxbb, std::sqrt(bsq));
	}
	//	rad = std::max(rad, r1);
      }
    }
  }
  //  maxeps = std::max(maxeps, maxvvar);

  if (maxeps > refdelta) return 1; // refinement; the region around the BH is never refined
  
  if ((maxeps < (0.5*refdelta)) || (maxbb < (0.5*refB)) || (maxden < (0.5*dfloor))) return -1;
      // || (maxden < (dfloor*0.5)) || (maxbb < (refB*0.5))) return -1; // derefinement
  // if (maxden > den2) return 1;
  // if (maxden < dfloor) return -1;
  return 0;
}

namespace{
  void psi_AB(Real r, Real cth, Real phi, Real *vr, Real *vth, Real *vphi){
    Real pl = cth, pl1 = 1., lharm = 1.;
    
    if((r>rmin)&&(r<rmax)){
    *vr = - (double)(lharm*(lharm+1)) * pl * (1.-r/rmin) * (rmax/r-1.) ;
    *vth =  ((rmin + rmax) / r - 2.) * (double)lharm * (cth * pl - pl1) * std::sqrt(1.-cth*cth);
    //(Rfun + b * cos(a * std::log(r/rmax)));
    }else{
      *vr = 0.;
      *vth = 0.;
    }
    *vphi = 0.;
  }

  
  void instar_interpolate(std::list<Real> instar_radius, std::list<Real> instar_lrho, std::list<Real> instar_lpress, std::list<Real> instar_mass, Real r2, Real rcore, Real *rho, Real *p){
    int nl =instar_radius.size();

    Real rho1, rho0=0., press1, press0 = 0., r0=instar_radius.front()*1.1, r1, m1, mcore=-1., pcore = -1., rhocore = 0., m0 = mstar;

    std::list<Real> instar_radius1 = instar_radius;
    std::list<Real> instar_lrho1 = instar_lrho;
    std::list<Real> instar_lpress1 = instar_lpress;
    std::list<Real> instar_mass1 = instar_mass;

    //    std::cout << "lrho: " << instar_lrho1.front() << std::endl;

    if (r2>instar_radius1.front()){
      rho1 = instar_lrho1.front(); // no conversion so far
      press1 = instar_lpress1.front(); // no conversion so far

      *rho = rho1; // (rho1-rho0)/(r1-r0)*(r2-r0)+rho0;
      *p = press1; // (press1-press0)/(r1-r0)*(r2-r0)+press0;
      return;
    }

    for(int k = 0; k<nl; k++){
      rho1 = instar_lrho1.front(); // no conversion so far
      press1 = instar_lpress1.front(); // no conversion so far
      r1 = instar_radius1.front();
      m1 = instar_mass1.front();

      // std::cout << instar_array.size() << std::endl;
      instar_lrho1.pop_front(); instar_lpress1.pop_front(); instar_radius1.pop_front(); instar_mass1.pop_front();
      if ((r2<r0)&&(r2>=r1)){
	//      std::cout << r0 << " > "<< r2 << " > " << r1 << std::endl;
	// std::cout << std::pow(10., (lrho1-lrho0)/(r1-r0)*(r2-r0)+lrho0) << std::endl;
	*rho = (rho1-rho0)/(r1-r0)*(r2-r0)+rho0;
	*p = (press1-press0)/(r1-r0)*(r2-r0)+press0;
	return;
	// return std::pow(10., (lrho1-lrho0)/(r1-r0)*(r2-r0)+lrho0);
      }
      rho0 = rho1;
      press0 = press1 ;
      r0 = r1;
    }
    *rho = rho1;
    *p = press1;
    return;
  }

}
