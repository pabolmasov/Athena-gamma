//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file quark.cpp
//  \brief star propagation test on a coarse grid

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
  Real four_pi_G;
  Real rstar; // xcore = rstarcore/rstar
  int xmin, xmax, magbeta; // multipolar structure in magnetic field or convection velocity field
Real vx, vy, vz; // transverse velocity
std::string starmodel; // the name of the MESA file containing the density and pressure profiles

Real rscale, mscale;

void instar_interpolate(std::list<Real> instar_radius, std::list<Real> instar_lrho, std::list<Real> instar_lpress, std::list<Real> instar_mass, Real r2, Real rres, Real *rho, Real *p);

}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // gam = pin->GetReal("hydro","gamma");

    four_pi_G = pin->GetReal("problem","four_pi_G");

    starmodel = pin->GetString("problem", "starmodel");

    rscale = pin->GetReal("problem","rscale");
    mscale = pin->GetReal("problem","mscale");
    // rhostar = pin->GetReal("problem","rhostar");
    // pstar = pin->GetReal("problem","pstar");
    // temp = pin->GetReal("problem","temp");

    rstar = pin->GetReal("problem","rstar");
    // rhalo = pin->GetReal("problem","rhalo");

    vx = pin->GetReal("problem","vx");
    vy = pin->GetReal("problem","vy");
    vz = pin->GetReal("problem","vz");
    // vxbgd = pin->GetReal("problem","vxbgd");
    // vybgd = pin->GetReal("problem","vybgd");
    // vzbgd = pin->GetReal("problem","vzbgd");

    SetFourPiG(four_pi_G);

  // AMR setup:
    //  if (adaptive==true)
 //   EnrollUserRefinementCondition(RefinementCondition);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
 //   Real x0 = 12.0/1024.0, y0 = 0.0, z0 = 0.0, r = 6.0/1024.0;
 
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
      // std::cout << tmp_m << " "<< tmp_r << " " << tmp_lrho << " " << tmp_lpress << std::endl ;
      instar_mass.push_back(tmp_m);
      instar_radius.push_back(tmp_r);
      instar_lrho.push_back(tmp_lrho);
      instar_lpress.push_back(tmp_lpress/G);
      if (tmp_r > instar_rmax) instar_rmax = tmp_r;
      if (tmp_m > instar_mtot) instar_mtot = tmp_m;
    }
    indata.close();
    
    Real rhostar, pstar, mtot = 0., minstar = 0.;
    
    instar_interpolate(instar_radius, instar_lrho, instar_lpress, instar_mass, rstar, 0., &rhostar, &pstar) ;
    
    Real alpha = std::sqrt(6./four_pi_G*pstar/SQR(rhostar)-SQR(rstar)/3.);
    
    Real thetastar = 1./std::sqrt(1.+SQR(rstar/alpha)/3.);
    
    std::cout << "alpha = " << alpha << std::endl;
    
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
          Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k);
          Real dx = pcoord->dx1v(i), dy = pcoord->dx2v(j), dz = pcoord->dx3v(k);
          Real rsq = SQR(x)+SQR(y)+SQR(z);
          Real r2 = std::sqrt(rsq);
          Real theta = 1./std::sqrt(1.+rsq/SQR(alpha)/3.);
          instar_interpolate(instar_radius, instar_lrho, instar_lpress, instar_mass, r2, 0., &dencurrent, &pcurrent) ; // * pcoord->dx1f(i));

          if(rsq > SQR(rstar)){
              // stellar halo
              dencurrent = rhostar * std::pow(theta/thetastar, 5.);
              pcurrent = pstar * std::pow(theta/thetastar, 6.);
          }
          
          phydro->w(IDN,k,j,i) = dencurrent;
          // pressure scales with M^2, hence we need to downscale it twice
          phydro->w(IPR,k,j,i) = pcurrent;

          if (NSCALARS > 0){
              constexpr int scalar_norm = NSCALARS > 0 ? NSCALARS : 1.0;
              Real d0 = phydro->w(IDN,k,j,i);
              if (rsq < SQR(rstar)){
                 for (int n=0; n<NSCALARS; ++n) {
                     pscalars->s(n,k,j,i) = d0*1.0/scalar_norm;
                     // pscalars->r(n,k,j,i) = 1.;
                     minstar += dencurrent * dx * dy * dz ;
                 }
               }
               else{
                 for (int n=0; n<NSCALARS; ++n) {
                   pscalars->s(n,k,j,i) = 0.;
                   //pscalars->r(n,k,j,i) = 0.0;
                 }
               }
          }
              
          mtot += dencurrent * dx * dy * dz ;
          
          // phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i);
         /* if (rsq < SQR(rhalo)){
            if (rsq < SQR(rstar)){
                  phydro->w(IDN,k,j,i) = rhostar ;
                  phydro->w(IPR,k,j,i) = SQR(rhostar-rho)*(SQR(rstar)-rsq) * 2./3. * G * PI + rho*temp; // pressure for rho=const
              }
              else{
                  phydro->w(IDN,k,j,i) = std::min(rho * SQR(rhalo) / rsq, rhostar * 0.01) ;
              }
          }
          */
              phydro->w(IM1,k,j,i) = vx;
              phydro->w(IM2,k,j,i) = vy;
              phydro->w(IM3,k,j,i) = vz;
      }
    }
  }

    std::cout << "Mtot = " << mtot << "(" << minstar << ")\n";
    
 // pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is, ie, js, je, ks, ke); //
    AthenaArray<Real> bb;
    bb.NewAthenaArray(3, ke+1, je+1, ie+1);
    peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);

}

namespace{

void instar_interpolate(std::list<Real> instar_radius, std::list<Real> instar_lrho, std::list<Real> instar_lpress, std::list<Real> instar_mass, Real r2, Real rcore, Real *rho, Real *p){
  
    int nl =instar_radius.size();

  Real rho1, rho0=0., press1, press0 = 0., r0=instar_radius.front()*1.1, r1, m1, mcore=-1., pcore = -1., rhocore = 0., m0 = 1.;

    Real r2n = r2/rscale;
    
  std::list<Real> instar_radius1 = instar_radius;
  std::list<Real> instar_lrho1 = instar_lrho;
  std::list<Real> instar_lpress1 = instar_lpress;
  std::list<Real> instar_mass1 = instar_mass;

  //    std::cout << "lrho: " << instar_lrho1.front() << std::endl;

  if (r2n>instar_radius1.front()){
      // we should not appear here
    rho1 = instar_lrho1.front(); // no conversion so far
    press1 = instar_lpress1.front(); // no conversion so far

      *rho = rho1 * mscale / std::pow(rscale, 3.); // (rho1-rho0)/(r1-r0)*(r2-r0)+rho0;
      *p = press1 * SQR(mscale/SQR(rscale));    // (press1-press0)/(r1-r0)*(r2-r0)+press0;
    return;
  }

  for(int k = 0; k<nl; k++){
    rho1 = instar_lrho1.front(); // no conversion so far
    press1 = instar_lpress1.front(); // no conversion so far
    r1 = instar_radius1.front();
    m1 = instar_mass1.front();

    // std::cout << instar_array.size() << std::endl;
    instar_lrho1.pop_front(); instar_lpress1.pop_front(); instar_radius1.pop_front(); instar_mass1.pop_front();
    if ((r2n<=r0)&&(r2n>=r1)){
  //      std::cout << r0 << " > "<< r2 << " > " << r1 << std::endl;
  // std::cout << std::pow(10., (lrho1-lrho0)/(r1-r0)*(r2-r0)+lrho0) << std::endl;
        *rho = (rho1-rho0)/(r1-r0)*(r2n-r0)+rho0;
        *p = (press1-press0)/(r1-r0)*(r2n-r0)+press0;
        *rho *= mscale / std::pow(rscale, 3.);
        *p *= SQR(mscale/SQR(rscale));
        return ;
  // return std::pow(10., (lrho1-lrho0)/(r1-r0)*(r2-r0)+lrho0);
    }
    rho0 = rho1;
    press0 = press1 ;
    r0 = r1;
  }
  *rho = rho1 * mscale / std::pow(rscale, 3.);
  *p = press1 * SQR(mscale/SQR(rscale));
  return;
}
}
