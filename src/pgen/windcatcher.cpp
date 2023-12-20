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
// #include <gsl/gsl_sf.h> // GNU scientific library
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
  Real four_pi_G, gam, cs;
  Real rper, tper, Mcoeff, rzero, mstar, bgdrho, temp, amp; // drcorona has the shape of Lane-Emden with n=5; outside rstar-drcorona, density and pressure behave according to LE5
  bool ifwind;

Real mdot, twind, overkepler; // wind parameters
Real drho;
}

Real addmass = 1.0e3, rgrav, rstar, BHgmax, stargmax, ecc=1.;

/*
int RefinementCondition(MeshBlock *pmb);

void Tracer(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
            const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
            AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
*/
Real MyTimeStep(MeshBlock *pmb);


Real true_anomaly(Real time);

void starcoord(Real time, Real* xstar, Real* ystar, Real* zstar, Real* vxstar, Real* vystar, Real* vzstar);

void gravs(MeshBlock *pmb, const Real time, const Real dt,
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

  // rzero = pin->GetReal("problem","rzero");
  rper = pin->GetReal("problem","rper");

  rstar = pin->GetReal("problem","rstar");
  mstar = pin->GetReal("problem","mstar");
    
  addmass = pin->GetReal("problem","mBH"); // black hole mass
  rgrav = addmass/1e3 * 2.1218; // GM_{\rm BH}/c^2
  BHgmax = addmass / SQR(rgrav); // GM/R^2 at R = 3GM/c^2
  stargmax = mstar / SQR(rstar) ;

  bgdrho = pin->GetReal("problem","bgdrho");
  temp = pin->GetReal("problem","temp");
    
    ecc = pin->GetReal("problem","ecc");
    if (ecc<1.0){
        Mcoeff = std::sqrt(addmass / 2.) * std::pow(rper/(1.-ecc), -1.5);
    }else{
        Mcoeff = std::sqrt(addmass / rper)/rper;
    }
    ifwind = pin->GetBoolean("problem","ifwind");
    overkepler = pin->GetReal("problem","overkepler");
    mdot = pin->GetReal("problem","mdot");
    twind = pin->GetReal("problem","twind");
    
    drho = mdot / 4./PI*3./(rstar*rstar*rstar);
    //  SetFourPiG(four_pi_G); // works without self-gravity??

  // 
  EnrollUserExplicitSourceFunction(gravs);
  //  EnrollUserExplicitSourceFunction(stargrav);
  EnrollUserTimeStepFunction(MyTimeStep);
  //  if (NSCALARS>0){
  //   EnrollUserExplicitSourceFunction(Tracer);
  // }

  // AMR setup:
//  if (adaptive==true)
  //  EnrollUserRefinementCondition(RefinementCondition);

}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for gravity from a binary
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
 //   Real x0 = 12.0/1024.0, y0 = 0.0, z0 = 0.0, r = 6.0/1024.0;
 
  Real m1 = addmass; // black hole mass (must be the same as additional mass in src/hydro/srcterms/self_gravity.cpp)

  Real bgdp = temp * bgdrho;
    Real G = 1.;
  
  Real rvir = G * m1 / temp;
  Real dencurrent = 0., pcurrent = 0.;
  
  std::int64_t iseed = -1 - gid;
    
    if(Globals::my_rank == 0) std::cout << "time limits " << std::sqrt(rgrav/BHgmax) << ", " << std::sqrt(rstar/stargmax) << " " << bgdrho / drho <<   "\n";
  
 // if(Globals::my_rank == 0)std::cout << "rho_c / rho_bgd = " << rhostar/bgdrho << "\n";
   // Real xstar, ystar, zstar, vxstar, vystar, vzstar;
//    starcoord(0., &xstar, &ystar, &zstar, &vxstar, &vystar, &vzstar)
    
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
          Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k);
          Real xf = pcoord->x1f(i), yf = pcoord->x2f(j), zf = pcoord->x3f(k);  

          Real r1 = std::sqrt(SQR(x)+SQR(y)+SQR(z)); // distance to the BH
          // Real r2 = std::sqrt(SQR(x-rzero)+SQR(y)+SQR(z)), r2f = std::sqrt(SQR(xf-rzero)+SQR(yf)+SQR(zf));
	  
         // Real cth2 = z / r2, phi2 = std::atan2(y,x-rzero);
         // Real cth2f = zf / r2f, phi2f = std::atan2(yf,xf-rzero);

          Real rhogas = bgdrho, pgas = bgdrho * temp;
          rhogas = bgdrho ; //* std::exp(std::max(std::min(rvir/std::sqrt(SQR(r1)), 30.), 1.0e-10));
          pgas = bgdrho * temp ; //* std::exp(std::max(std::min(rvir/std::sqrt(SQR(r1)), 30.), 1.0e-10)) * temp;
          Real dx = pcoord->dx1f(i), dy = pcoord->dx2f(j), dz = pcoord->dx3f(k);
          Real dr = std::sqrt(SQR(dx)+SQR(dy)+SQR(dz)), dv = std::fabs(dx*dy*dz);
	  	  
	  phydro->w(IDN,k,j,i) = rhogas;
	  // pressure scales with M^2, hence we need to downscale it twice
	  if (NON_BAROTROPIC_EOS) {
	    phydro->w(IPR,k,j,i) = pgas ; // std::max(pcurrent,pgas);
	  }

      Real randvel=0.;
      phydro->w(IM3,k,j,i) = 0.0 ;
	  phydro->w(IM1,k,j,i) = 0.0;
	  phydro->w(IM2,k,j,i) = 0.0;
	  
      }
    }
  }

    AthenaArray<Real> bb;
    bb.NewAthenaArray(3, ke+2*NGHOST+1, je+2*NGHOST+1, ie+2*NGHOST+1);
    peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);

}

/*
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
*/

 Real MyTimeStep(MeshBlock *pmb)
 {
   Real min_dt;

     min_dt = std::min(std::sqrt(rgrav/BHgmax), std::sqrt(rstar/stargmax)) * 0.01;
     
     if (drho > 0.) min_dt = std::min(bgdrho/drho * 0.1, min_dt);
     /*
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
      */
     
   return min_dt;
 }

Real gfun(Real x, Real y, Real z, Real gmax, Real rmin){
  // black hole gravitational potential
  Real r = std::sqrt(SQR(x)+SQR(y)+SQR(z));

    return std::min(SQR(rmin/r), r/rmin) * gmax;
    /*
  if (r>=(3.*rgrav)){
    return gmax / SQR(r/rgrav-2.); // addmass/SQR(r-2.*rgrav);
  }
  else{
    return (r/rgrav) / 3. * gmax; // addmass/SQR(rgrav);
  }
     */
}

Real true_anomaly(Real time){
    Real M = Mcoeff * (time - tper); // global Mcoeff is sqrt(GM/Rp^3), global tper is the time of pericenter passage
    if (ecc >= 1.){
        // parabolic (hyperbolic not included yet!)
        return 2. * std::atan(2.*std::sinh(std::asinh(1.5*M)/3.));
    }
    else{
        // elliptic
        Real E = M , E1 = 0., tol = 1e-8; // eccentric anomaly, M = E - e sin(E)
        while(std::abs(E-E1)>tol){
            E = E1;
            E1 = M + ecc * std::sin(E);
        }
        Real beta = ecc / (1.+std::sqrt(1.-SQR(ecc)));
        return E + 2.*std::atan(beta * std::sin(E)/(1.-beta*std::sin(E)));
        //          return 2.*std::atan(std::sqrt((1.+ecc)/(1.-ecc)*std::tan(E/2.)));
    }
}


void starcoord(Real time, Real* xstar, Real* ystar, Real* zstar, Real* vxstar, Real* vystar, Real* vzstar){
    // parabolic motion with pericenter distance rper, true anomaly nu
    Real nu = true_anomaly(time);
    Real rad = (1.0+ecc) * rper / (1.0+ecc * std::cos(nu));
    
    Real cosnu = std::cos(nu), sinnu = std::sin(nu);;
    
    *xstar = rad * cosnu;
    *ystar = rad * sinnu;
    *zstar = 0.;
    
    Real v = std::sqrt(2.*addmass/rad);
    
    *vxstar = - v * sinnu / (1.+cosnu)/2.; // ecc=1 only! TODO: add correct expressions for elliptic curves
    *vystar = v / 2.;
    *vzstar = 0.;
}


void gravs(MeshBlock *pmb, const Real time, const Real dt,
           const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
           const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
           AthenaArray<Real> &cons_scalar){
    
    Real xstar, ystar, zstar, vxstar, vystar, vzstar;
    starcoord(time+dt/2., &xstar, &ystar, &zstar, &vxstar, &vystar, &vzstar);
    
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
        Real z = pmb->pcoord->x3v(k);
        for (int j=pmb->js; j<=pmb->je; ++j) {
            Real y = pmb->pcoord->x2v(j);
            for (int i=pmb->is; i<=pmb->ie; ++i) {
                Real x = pmb->pcoord->x1v(i);
                
                Real x1 = x-xstar, y1 = y-ystar, z1 = z-zstar;
                Real rsqeffstar = SQR(x1)+SQR(y1)+SQR(z1);
                Real r1 = std::sqrt(rsqeffstar);
                //  r1 = std::max(r1, rstar*0.01);
                
                if((r1 <= rstar) && (ifwind==true)){
                    // Real mrate = mdot / 4./PI*3./(rstar*rstar*rstar);
                    Real vwind = std::sqrt(2.*mstar/rstar * overkepler + 0.5 * (SQR(rstar)-SQR(r1))*stargmax/rstar);
                    
                    cons(IDN,k,j,i) += drho * dt; // mass created within the sphere
                    cons(IM1,k,j,i) += drho * (vwind * x1/r1 + vxstar) * dt ;
                    cons(IM2,k,j,i) += drho * (vwind * y1/r1 + vystar) * dt ;
                    cons(IM3,k,j,i) += drho * (vwind * z1/r1 + vzstar) * dt ;
                    if (NON_BAROTROPIC_EOS) {
                        cons(IEN, k, j, i) += drho * (twind + (SQR(vwind)+SQR(vxstar)+SQR(vystar)+2.*(x1 * vxstar+y1*vystar+z1*vzstar)*vwind/r1)/2.) * dt ;
                    }
                }
                
                Real rsqeff = SQR(x)+SQR(y)+SQR(z);
                Real reff = std::sqrt(rsqeff), fadd = gfun(x,y,z, BHgmax, rgrav), fstar = gfun(x1, y1, z1, stargmax, rstar), den = prim(IDN,k,j,i);
                
                Real g1 = fadd * (x/reff) + fstar * (x1/r1) ; // (BHphifun(pmb->pcoord->x1v(i+1), y, z)-BHphifun(pmb->pcoord->x1v(i-1), y, z))/2.,
                Real g2 = fadd * (y/reff) + fstar * (y1/r1); // (BHphifun(x,pmb->pcoord->x2v(j+1), z)-BHphifun(x, pmb->pcoord->x2v(j-1), z))/2.,
                Real g3 = fadd * (z/reff) + fstar * (z1/r1); // (BHphifun(x,y,pmb->pcoord->x3v(k+1))-BHphifun(x, y, pmb->pcoord->x3v(k-1)))/2.;
                
                cons(IM1,k,j,i) -=  ( g1 * dt ) * den ; //dtodx1 * den * (BHphifun(pmb->pcoord->x1v(i+1), y, z)-BHphifun(pmb->pcoord->x1v(i-1), y, z))/2.;
                cons(IM2,k,j,i) -=  ( g2 * dt ) * den ; // dtodx2 * den * (BHphifun(x,pmb->pcoord->x2v(j+1), z)-BHphifun(x, pmb->pcoord->x2v(j-1), z))/2.;
                cons(IM3,k,j,i) -=  ( g3 * dt ) * den ; // dtodx3 * den * (BHphifun(x,y,pmb->pcoord->x3v(k+1))-BHphifun(x, y, pmb->pcoord->x3v(k-1)))/2.;

                if (NON_BAROTROPIC_EOS) {
                    cons(IEN,k,j,i) -= (g1 * prim(IM1, k,j,i) + g2 * prim(IM2,k,j,i) +  g3 * prim(IM3, k,j,i)) * dt * den;
                }
            }
        }
    }
}

namespace{


}
