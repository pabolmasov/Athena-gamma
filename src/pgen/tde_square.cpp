//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sg_tde.cpp
//  \brief Problem generator  for a self-gravitating star tidal disruption 

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
#include <cfloat>
using std::ifstream;
#include <limits>

// #include <fftw3.h>
#include <gsl/gsl_sf.h> // GNU scientific library
// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../fft/athena_fft.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/mg_gravity.hpp"
// #include "../gravity/fft_gravity.hpp"
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

/*

#if SELF_GRAVITY_ENABLED != 2
#error "This problem generator requires Multigrid gravity solver."
#endif


#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

*/

namespace{
  Real four_pi_G, gam;
  Real rper, rzero, rstar, mstar, bgdrho, temp, overkepler, drcorona, rhalo, amp; // drcorona has the shape of Lane-Emden with n=5; outside rstar-drcorona, density and pressure behave according to LE5
  bool usetracer, iftab, ifisostar; 
  //  Real denpro(Real x); // density profile inside the star, normalised by unity; argument is the distance from the centre/ R*
  //  Real ppro(Real x); // pressure profile (consistent with denpro)
  // ifstream indata;
  std::string starmodel; // the name of the LE solution or MESA file containing the density and pressure profiles
  Real rscale, mscale;
  void instar_interpolate(std::list<Real> instar_radius, std::list<Real> instar_lrho, std::list<Real> instar_lpress, std::list<Real> instar_mass, Real r2, Real rres, Real *rho, Real *p);
  // Real instar_interpolate(std::list<Real> instar_radius, std::list<Real>instar_array, std::list<Real> instar_mass, Real r2, Real rres);
  Real psi_AB(Real r, Real cth, Real phi);
  Real azfun(Real r, Real z);
  Real bbgd_x, bbgd_y, bbgd_z; // background uniform magnetic field

  int lharm, mharm, rmin, rmax; // multipolar structure in magnetic field or convection velocity field
  Real qx, qy, qz; // perturbation wave vector
  bool randnoise; // if the noise is white rather than a single wave 

  Real cs;

  Real isostar(Real r);

}

Real refden, refB, threshold, magbeta, refR, addmass = 1.0e3, rBHsq = 100., bgdtemp, rgrav, BHgmax;

int RefinementCondition(MeshBlock *pmb);

void Tracer(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
            const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
            AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);

Real MyTimeStep(MeshBlock *pmb);

void BHgrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // gam = pin->GetReal("hydro","gamma");
  if (!NON_BAROTROPIC_EOS) {// isothermal case                                                            
    cs = pin->GetReal("hydro","iso_sound_speed");
  }
  else{
    gam = pin->GetReal("hydro","gamma");        
  }
  four_pi_G = pin->GetReal("problem","four_pi_G");

  ifisostar = pin->GetBoolean("problem", "ifisostar");

  //  rscale = pin->GetReal("problem","rscale");

  rzero = pin->GetReal("problem","rzero");
  rper = pin->GetReal("problem","rper");

  rstar = pin->GetReal("problem","rstar");
  // rhalo = pin->GetReal("problem","rhalo");
  rmin = pin->GetReal("problem","rmin"); // the magnetized layer
  rmax = pin->GetReal("problem","rmax");

  magbeta = pin->GetReal("problem","magbeta"); // magnetic parameter

  addmass = pin->GetReal("problem","mBH"); // black hole mass
  rgrav = addmass/1e3 * 2.1218; // GM_{\rm BH}/c^2                                                                                                                 
  BHgmax = addmass / SQR(rgrav); // GM/R^2 at R = 3GM/c^2
  // rBHsq = SQR(pin->GetReal("problem","rBH")); // softening the inner part of the potential

  //    xcore = rstarcore / rstar;
  mscale = mstar = pin->GetReal("problem","mstar"); // note that addmass should be adjusted in src/hydro/srcterms/self_gravity.cpp as well
  // fcore = pin->GetReal("problem","fcore");
  //  dflat = pin->GetBoolean("problem", "dflat");

  bgdrho = pin->GetReal("problem","bgdrho");
  temp = pin->GetReal("problem","temp");   
  bgdtemp = temp; // characteristic background temperature (used by refinement condition)

  bbgd_x = pin->GetReal("problem","bgdBx");
  bbgd_y = pin->GetReal("problem","bgdBy");
  bbgd_z = pin->GetReal("problem","bgdBz");

  usetracer = pin->GetBoolean("problem", "usetracer");
  refden = pin->GetReal("problem","refden");
  refB = pin->GetReal("problem","refB");
  refR = pin->GetReal("problem","refR"); // filter for passive scalar: regions inside the star are resolved
  threshold = pin->GetReal("problem","thresh");
  
  // temp = pin->GetReal("problem","temp");
  amp = pin->GetReal("problem","amp");
  lharm = pin->GetReal("problem","lharm"); // if lharm <= 0, the field is considered toroidal
  mharm = pin->GetReal("problem","mharm");
  
  randnoise = pin->GetBoolean("problem", "randnoise");

  qx =  pin->GetReal("problem","qx");
  qy =  pin->GetReal("problem","qy");
  qz =  pin->GetReal("problem","qz");

  //  gstepnumber = pin->GetReal("problem","gstepnumber");

  overkepler = std::sqrt(2.*rper/rzero);

  SetFourPiG(four_pi_G); // works without self-gravity??

  // 
  EnrollUserExplicitSourceFunction(BHgrav);
  //  EnrollUserTimeStepFunction(MyTimeStep);
  //  if (NSCALARS>0){
  //   EnrollUserExplicitSourceFunction(Tracer);
  // }

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
 
  Real m1 = addmass; // black hole mass (must be the same as additional mass in src/hydro/srcterms/self_gravity.cpp)

  // Real drsq = rBHsq;
  //  Real rhalo = 5. *rstar;

  Real bgdp = temp * bgdrho;
  Real G = four_pi_G / (4.0 * PI);
  
  Real rvir = G * m1 / temp;
  Real dencurrent = 0., pcurrent = 0.;
  
  std::int64_t iseed = -1 - gid;
  
  Real rnorm, rcsnorm=0., starmass_empirical = 0., totalmass_empirical = 0.;

  Real tmp_m, tmp_r, tmp_lrho, tmp_lpress, instar_rmax = 0., instar_mtot, mcheck = 0.;

  Real rhostar, pstar, alpha, thetastar;
   
  alpha = rstar;
  rhostar = mstar / (4.0 * PI * std::sqrt(3.)) * std::pow(alpha, -3.0);
  if (ifisostar) rhostar = mstar / (4.0 * PI) * std::pow(alpha, -3.0);
  pstar = four_pi_G * SQR(rhostar * alpha) / 6. ;
  thetastar = 1.;
  if (!NON_BAROTROPIC_EOS) {// isothermal case
    Real mcorr = 0.9054779490719335;
    rhostar = mstar / (4.0 * PI * mcorr) * std::pow(alpha, -3.0);
    pstar = rhostar * SQR(cs);
  }
  
  if(Globals::my_rank == 0)std::cout << "rho_c / rho_bgd = " << rhostar/bgdrho << "\n";
  //  Real thetastar = 1./std::sqrt(1.+SQR(rstar/alpha)/3.);
  
  if (magbeta > 0.){ // magbeta <0 would turn off magnetic fields
    // interpolation for MF normalization
    rnorm = (rmax+rmin)/2.;

    rcsnorm = std::sqrt(2. * bgdrho * temp / magbeta);   
    if(Globals::my_rank == 0) std::cout << "norm = " << rcsnorm << std::endl;
  }

  AthenaArray<Real> ax, ay, az;
  //    AthenaArray<Real> ax, ay, az;
  int nx1 = block_size.nx1 + 2*NGHOST;
  int nx2 = block_size.nx2 + 2*NGHOST;
  int nx3 = block_size.nx3 + 2*NGHOST;
  ax.NewAthenaArray(nx3, nx2, nx1);
  ay.NewAthenaArray(nx3, nx2, nx1);
  az.NewAthenaArray(nx3, nx2, nx1);  
  
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
          Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k);
	  Real xf = pcoord->x1f(i), yf = pcoord->x2f(j), zf = pcoord->x3f(k);  
	  ax(k,j,i) = ay(k,j,i) = az(k,j,i) = 0.;
	  Real r1 = std::sqrt(SQR(x)+SQR(y)+SQR(z)); // distance to the BH
          Real r2 = std::sqrt(SQR(x-rzero)+SQR(y)+SQR(z)), r2f = std::sqrt(SQR(xf-rzero)+SQR(yf)+SQR(zf));
	  
	  Real theta = 1./std::sqrt(1.+SQR(r2/alpha)/3.);
	  Real cth2 = z / r2, phi2 = std::atan2(y,x-rzero);
    Real cth2f = zf / r2f, phi2f = std::atan2(yf,xf-rzero);

    Real rhogas = bgdrho, pgas = bgdrho * temp;
	  rhogas = bgdrho * std::exp(std::max(std::min(rvir/std::sqrt(SQR(r1)), 30.), 1.0e-10));
	  pgas = bgdrho * std::exp(std::max(std::min(rvir/std::sqrt(SQR(r1)), 30.), 1.0e-10)) * temp;
	  Real dx = pcoord->dx1f(i), dy = pcoord->dx2f(j), dz = pcoord->dx3f(k);
	  Real dr = std::sqrt(SQR(dx)+SQR(dy)+SQR(dz)), dv = std::fabs(dx*dy*dz);
	  	  
	  if (ifisostar){
	    dencurrent = rhostar * isostar(r2/alpha);
	    pcurrent = pstar * isostar(r2/alpha);
	  }
	    else{
	      // n=5 solution
	      dencurrent = rhostar * std::pow(theta/thetastar, 5.);
	      pcurrent = pstar * std::pow(theta/thetastar, 6.);
	    }
	  
	  if (!NON_BAROTROPIC_EOS) {
	    dencurrent = rhostar * isostar(r2/alpha);
	    pcurrent = dencurrent * SQR(cs) ;
	  }

	  // 	  if (r2 <= rstar){ // inside the star
	  //   instar_interpolate(instar_radius, instar_lrho, instar_lpress, instar_mass, r2 / rstar, dr / rstar, &dencurrent, &pcurrent) ; // * pcoord->dx1f(i));
	  totalmass_empirical += dencurrent * dv ;
	  if((magbeta > 0.) && MAGNETIC_FIELDS_ENABLED){
	    if (lharm <= 0){ // toroidal field
	      Real rc = std::sqrt(SQR(xf-rzero)+SQR(yf)); // cylindric radius
	      // az(k,j,i) = azfun(rc, zf);
	      // vector potential at the boundaries:
	      for(int di = -1; di<=1; di++){
		for(int dj = -1; dj<=1; dj++){
		  for(int dk = -1; dk<=1; dk++){
		    Real xf1, yf1, zf1;
		    if (di == -1)xf1 = xf - dx;
                    if (di == 0)xf1 = xf;
                    if (di == 1)xf1 = xf + dx;
                    if (dj == -1)yf1 = yf - dy;
                    if (dj == 0)yf1 = yf;
                    if (dj == 1)yf1 = yf + dy;
                    if (dk == -1)zf1 = zf - dz;
                    if (dk == 0)zf1 = zf;
                    if (dk == 1)zf1 = zf + dz;
		    //                    Real yf1 = yf +(Real)dj * dy;
                    // Real zf1 = zf +(Real)dk * dz;
		    rc = std::sqrt(SQR(xf1-rzero)+SQR(yf1));
		    az(k+dk,j+dj,i+di) = azfun(rc, zf1);
		  }
		}
	      }	      
	    }
	    else{
	      Real aphi = psi_AB(r2f, cth2f, phi2f);
	      // cth2 * std::max((1. - r2 / rmax) * (r2 / rmin - 1. ), 0.) * rcsnorm ;
	      // ax(k,j,i) = -aphi * std::sin(phi2f) ;
	      // ay(k,j,i) = aphi * std::cos(phi2f) ;
              for(int di = -1; di<=1; di++){
                for(int dj = -1; dj<=1; dj++){
                  for(int dk = -1; dk<=1; dk++){
                    // Real xf1 = xf + (Real)di * dx;
                    // Real yf1 = yf + (Real)dj * dy;
                    // Real zf1 = zf + (Real)dk * dz;
                      Real xf1, yf1, zf1;
                      if (di == -1)xf1 = xf - dx;
                      if (di == 0)xf1 = xf;
                      if (di == 1)xf1 = xf + dx;
                      if (dj == -1)yf1 = yf - dy;
                      if (dj == 0)yf1 = yf;
                      if (dj == 1)yf1 = yf + dy;
                      if (dk == -1)zf1 = zf - dz;
                      if (dk == 0)zf1 = zf;
                      if (dk == 1)zf1 = zf + dz;

                      r2f = std::sqrt(SQR(xf1-rzero)+SQR(yf1)+SQR(zf1));
                    
                      cth2f = zf1 / r2f;
                      phi2f = std::atan2(yf1,xf1-rzero);

                      aphi = psi_AB(r2f, cth2f, phi2f);

                      ax(k+dk,j+dj,i+di) = -aphi * std::sin(phi2f);
                      ay(k+dk,j+dj,i+di) = aphi * std::cos(phi2f);
                  }
                }
              }
	    }
	  }
	  
	  Real randwave = std::cos(qx * x + qy * y + qz * z); // perturbation in the form of a flat wave
	  if (randnoise == true) randwave = ran2(&iseed) - 0.5;
	  phydro->w(IDN,k,j,i) = std::max(dencurrent * (amp * randwave+1.),rhogas);
	  // pressure scales with M^2, hence we need to downscale it twice
	  if (NON_BAROTROPIC_EOS) {
	    phydro->w(IPR,k,j,i) = pgas ; // std::max(pcurrent,pgas);
	  }

	  if (NSCALARS>0){
	    constexpr int scalar_norm = NSCALARS > 0 ? NSCALARS : 1.0;
	    // pscalars->r(0,k,j,i) = (Real)(r2<rstar);
	    Real d0 = phydro->w(IDN,k,j,i);
	    if (r2<rstar){
	      for (int n=0; n<NSCALARS; ++n) {
		pscalars->s(n,k,j,i) = d0*1.0/scalar_norm;
		starmass_empirical += dencurrent * dv ;
		//		pscalars->r(n, k, j, i) = 1.0;
	      }
	    }
	    else{
	      for (int n=0; n<NSCALARS; ++n) {
		pscalars->s(n,k,j,i) = d0*0.0/scalar_norm;
		// pscalars->r(n,k,j,i) = 0.0;
	      }
	      //	      pscalars->s(0,k,j,i) = 0.;
	    }
	  }

	  // phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i);
	  // velocities:
	  //	  phydro->w(IM1,k,j,i) = 0.;
	  //cphydro->w(IM2,k,j,i) = 0.;
	  Real randvel, cs;
	  if (randnoise == true){
	    randvel = ran2(&iseed) - 0.5;
	    cs = std::sqrt(pcurrent/(dencurrent+rhogas));
	  }
	  else{
	    randvel = 0.;
	  }
          phydro->w(IM3,k,j,i) =  cs * amp * randvel * 0.0 ;
	  phydro->w(IM1,k,j,i) = -std::sqrt(2.*G*m1 /rzero * (1.- rper/rzero));
	  phydro->w(IM2,k,j,i) = std::sqrt(G*m1/rzero) * overkepler;
	  
      }
    }
  }

  //  MPI_Allreduce(MPI_IN_PLACE, &totalmass_empirical, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  // MPI_Allreduce(MPI_IN_PLACE, &starmass_empirical, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  // std::cout << "empirical M* = " << starmass_empirical << "\n";
  // std::cout << "empirical M = " << totalmass_empirical << "\n";

  if(MAGNETIC_FIELDS_ENABLED){
    // initialize interface B
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
	for (int i=is; i<=ie+1; i++) {
	  if(magbeta > 0.){
	    pfield->b.x1f(k,j,i) = ((az(k,j+1,i) - az(k,j,i))/pcoord->dx2f(j) -
				    (ay(k+1,j,i) - ay(k,j,i))/pcoord->dx3f(k)) * rcsnorm + bbgd_x * std::sqrt(bgdrho*temp);
	  }
	  else{
	    pfield->b.x1f(k,j,i) = 0.;
	  }
	}
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
	for (int i=is; i<=ie; i++) {
          if(magbeta > 0.){
	    pfield->b.x2f(k,j,i) = ((ax(k+1,j,i) - ax(k,j,i))/pcoord->dx3f(k) -
				    (az(k,j,i+1) - az(k,j,i))/pcoord->dx1f(i)) * rcsnorm 
	      + bbgd_y * std::sqrt(bgdrho*temp);
	  }
          else{
            pfield->b.x2f(k,j,i) = 0.;
          }
	}
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
	for (int i=is; i<=ie; i++) {
	  if(magbeta > 0.){
	    pfield->b.x3f(k,j,i) = ((ay(k,j,i+1) - ay(k,j,i))/pcoord->dx1f(i) -
				    (ax(k,j+1,i) - ax(k,j,i))/pcoord->dx2f(j)) * rcsnorm 
	      + bbgd_z * std::sqrt(bgdrho*temp);
	  }
	  else{
	    pfield->b.x3f(k,j,i) = 0.;
	  }
	}
      }
    }
  }

    // Calculate cell-centered magnetic field
  if(MAGNETIC_FIELDS_ENABLED){                                                                                                                    
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST);
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie, js, je, ks, ke);
  }
  else{
    AthenaArray<Real> bb;
    bb.NewAthenaArray(3, ke+2*NGHOST+1, je+2*NGHOST+1, ie+2*NGHOST+1);
    peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);
    // peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST);
  }
    /*
    if(MAGNETIC_FIELDS_ENABLED){
      pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is, ie, js, je, ks, ke); 
    }
    */
    //  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);

}

int RefinementCondition(MeshBlock *pmb)
{

  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> &R = pmb->pscalars->r; // scalar (=1 inside the star, =0 outside)
                                                                                    
  Real maxeps = 0.0, maxden = 0.0, maxR0 = 0.0;
  Real rad = 0.;

  Real rBH = 20. * rgrav ; // central region, remaining unresolved

  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {
        Real x = pmb->pcoord->x1v(i), y = pmb->pcoord->x2v(j), z = pmb->pcoord->x3v(k);
        //        Real xf = pmb->pcoord->x1f(i), yf = pmb->pcoord->x2f(j), zf = pmb->pcoord->x3f(k);
        Real r1 = std::sqrt(SQR(x)+SQR(y)+SQR(z)); // distance to the BH

        Real eps = 0., epsp = 0., den = 0., bb=0.,  bsq = 0., epsm = 0.;

        maxden = w(IDN,k,j,i);
        if (usetracer == true){
          if((R(0, k, j, i) > refR) && (r1 > rBH)){
            eps = std::sqrt(SQR(w(IDN,k,j,i+1)-w(IDN,k,j,i-1))
                            +SQR(w(IDN,k,j+1,i)-w(IDN,k,j-1,i))
                            +SQR(w(IDN,k+1,j,i)-w(IDN,k-1,j,i)))/(den+refden);
          }
        }
        maxeps = std::max(maxeps, eps);
        maxden = std::max(maxden, den);
        if (usetracer == true) maxR0 = std::max(maxR0, R(0, k, j, i));
      }
    }
  }

  if (maxeps > threshold) return 1; // refinement

  if(usetracer == true){
    if ((maxeps < (0.25*threshold)) || (maxR0 < (0.25*refR))) return -1; // derefinement
  }
  else{
    if ((maxeps < (0.25*threshold)) || (maxden < (0.25*refden))) return -1;
  }
}



 Real MyTimeStep(MeshBlock *pmb)
 {
   Real min_dt=FLT_MAX;

   Real rBHsq = 4., gstepnumber = 1e5;

   AthenaArray<Real> &w = pmb->phydro->w;
   AthenaArray<Real> &R = pmb->pscalars->r; 

   for (int k=pmb->ks; k<=pmb->ke; ++k) {
     for (int j=pmb->js; j<=pmb->je; ++j) {
       for (int i=pmb->is; i<=pmb->ie; ++i) {
	 Real r0 = R(0, k, j, i);
	 Real x = pmb->pcoord->x1v(i), y = pmb->pcoord->x2v(j), z = pmb->pcoord->x3v(k);
         // Real dx = pmb->pcoord->dx1v(i), dy = pmb->pcoord->dx2v(j), dz = pmb->pcoord->dx3v(k);

	 // Real dr = std::min(dx, std::min(dy, dz));
	 Real r1sq = SQR(x)+SQR(y)+SQR(z); // distance to the BH
	 // Real v =  std::sqrt(SQR(w(IM1, k, j, i))+SQR(w(IM2, k, j, i))+SQR(w(IM3, k, j, i))) * r1sq / addmass; // velocity
	 //dtG = std::sqrt((r1sq + rBHsq) / addmass) * dr ; // external gravity
	 // dtR = dr * () / std::sqrt(SQR(w(IM1, k, j, i))+SQR(w(IM2, k, j, i))+SQR(w(IM3, k, j, i))+w(IEN, k, j, i)/w(IDN, k, j, i));
	 Real csq = gam * w(IPR,k,j,i) / w(IDN,k,j,i);
	 Real vabs = std::sqrt(std::max(SQR(w(IM1,k,j,i)) + SQR(w(IM2,k,j,i)) + SQR(w(IM3,k,j,i)), csq));
	 Real dtG = csq / vabs  * ((r1sq + rBHsq) / addmass) * gstepnumber;

	 if (r0 > refR)min_dt = std::min(min_dt, dtG); // std::min(dtR, dtG));
       }
     }
   }
   return min_dt;
 }


Real BHgfun(Real x, Real y, Real z){
  // black hole gravitational potential                                                                                                                           
  Real r = std::sqrt(SQR(x)+SQR(y)+SQR(z));

  if (r>=(3.*rgrav)){
    return BHgmax / SQR(r/rgrav-2.); // addmass/SQR(r-2.*rgrav);                                                                                                  
  }
  else{
    return (r/rgrav) / 3. * BHgmax; // addmass/SQR(rgrav);                                                                                                        
  }
}

void BHgrav(MeshBlock *pmb, const Real time, const Real dt,
	    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
	    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
	    AthenaArray<Real> &cons_scalar){
  
  //  bool barycenter = false;

  //  Real dtodx1 = dt / pmb->pcoord->dx1v(0), dtodx2 = dt / pmb->pcoord->dx2v(0), dtodx3 = dt / pmb->pcoord->dx3v(0); // relies on the uniformity of the grid!!!

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real z = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real y = pmb->pcoord->x2v(j);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real x = pmb->pcoord->x1v(i);

          Real rsqeff = SQR(x)+SQR(y)+SQR(z);
          Real reff = std::sqrt(rsqeff), fadd = BHgfun(x,y,z), den = prim(IDN,k,j,i);

          Real g1 = fadd * (x/reff) ; // (BHphifun(pmb->pcoord->x1v(i+1), y, z)-BHphifun(pmb->pcoord->x1v(i-1), y, z))/2.,
          Real g2 = fadd * (y/reff) ; // (BHphifun(x,pmb->pcoord->x2v(j+1), z)-BHphifun(x, pmb->pcoord->x2v(j-1), z))/2.,
          Real g3 = fadd * (z/reff) ; // (BHphifun(x,y,pmb->pcoord->x3v(k+1))-BHphifun(x, y, pmb->pcoord->x3v(k-1)))/2.;

          cons(IM1,k,j,i) -=  ( g1 * dt ) * den ; //dtodx1 * den * (BHphifun(pmb->pcoord->x1v(i+1), y, z)-BHphifun(pmb->pcoord->x1v(i-1), y, z))/2.;
          cons(IM2,k,j,i) -=  ( g2 * dt ) * den ; // dtodx2 * den * (BHphifun(x,pmb->pcoord->x2v(j+1), z)-BHphifun(x, pmb->pcoord->x2v(j-1), z))/2.;
          cons(IM3,k,j,i) -=  ( g3 * dt ) * den ; // dtodx3 * den * (BHphifun(x,y,pmb->pcoord->x3v(k+1))-BHphifun(x, y, pmb->pcoord->x3v(k-1)))/2.;

	  //fadd * (z / std::sqrt(rsqeff)) ;
	if (NON_BAROTROPIC_EOS) {
	  cons(IEN,k,j,i) -= (g1 * prim(IM1, k,j,i) + g2 * prim(IM2,k,j,i) +  g3 * prim(IM3, k,j,i)) * dt * den;
	}
      }
    }
  }
}

namespace{
  Real psi_AB(Real r, Real cth, Real phi){
    if ((r< rmin) || (r>rmax)){
      return 0.;
    }
    Real pl = gsl_sf_legendre_Pl(lharm, cth), pl1 = 0.;
    if (lharm > 0) pl1 = gsl_sf_legendre_Pl(lharm-1, cth);

    return (cth * pl - pl1) * std::max((r/rmin-1.) * (rmax/r-1.), 0.) ;
  }
  
  Real azfun(Real r, Real z){ // Atheta(r) for a toroidal loop
    Real rmid = (rmax+rmin)/2., rthick = (rmax-rmin)/2.;

    if(fabs(z)>rthick){
      return 0.;
    }
    
    Real R1 = rmid - std::sqrt(std::max(SQR(rthick)-SQR(z), 0.)), R2 = rmid + std::sqrt(std::max(SQR(rthick)-SQR(z), 0.));
    
    if(r <= R1){
      return 0.;
    }

    if (r >= R2){
      return (R2-R1); // /(rmax-rmin);
    }
    else{
      return (r-R1); // /(rmax-rmin);
    }
  }

  Real isostar(Real r){
    Real r0 = 2.312 ; // from Raga et al (2013)
    if (r < r0){
      // near field
      Real c = 0.548; // 3.-std::sqrt(6.) \simeq 0.551
      return (1.+(2.*c-1.) * SQR(r))/SQR(1.+c*SQR(r));
    }else{
      Real q = 0.735 / std::sqrt(r) * (1.+5.08 * std::pow(r, -1.94))
	* std::cos(std::sqrt(7.)/2. * std::log(r) + 5.396 * (1. + 0.92 * std::pow(r, -2.31)));
      return (1.+q)/3./SQR(r);
    }
  }


}
