//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gtest.cpp
//  \brief star in an external uniform gravity field

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstring>    // memset
#include <ctime>
#include <iomanip>
#include <iostream>
#include <cfloat>
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

/*
#if SELF_GRAVITY_ENABLED != 2
#error "This problem generator requires Multigrid gravity solver."
#endif
*/

namespace{
Real four_pi_G;
Real vx, vy, vz; // transverse velocity

Real rscale, mscale;

Real bgdrho, bgdpress;

Real cs;

Real isostar(Real r);

Real xstar, ystar, zstar;
}

int RefinementCondition(MeshBlock *pmb);

Real MyTimeStep(MeshBlock *pmb);

void grav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void psmear(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

Real geff, refR, threshold, gam, dcoeff, slimit, gstepnumber;

void Mesh::InitUserMeshData(ParameterInput *pin) {
    if(NON_BAROTROPIC_EOS){
        gam = pin->GetReal("hydro","gamma");
        cs = std::sqrt( gam * bgdpress / bgdrho) ;
    }
    else{
        cs = pin->GetReal("hydro","iso_sound_speed");
    }

    four_pi_G = pin->GetReal("problem","four_pi_G");
    
    geff = pin->GetReal("problem","geff"); // external gravity
    
    rscale = pin->GetReal("problem","rscale");
    mscale = pin->GetReal("problem","mscale");

    xstar = pin->GetReal("problem","x0");
    ystar = pin->GetReal("problem","y0");
    zstar = pin->GetReal("problem","z0");

    vx = pin->GetReal("problem","vx");
    vy = pin->GetReal("problem","vy");
    vz = pin->GetReal("problem","vz");
    
    bgdrho = pin->GetReal("problem","bgdrho");
    bgdpress = pin->GetReal("problem","bgdpress");

    dcoeff = pin->GetReal("problem","dcoeff");
    slimit = pin->GetReal("problem","slimit");
    gstepnumber = pin->GetReal("problem","gstepnumber");

    SetFourPiG(four_pi_G);

   EnrollUserExplicitSourceFunction(grav);
  //  EnrollUserExplicitSourceFunction(psmear);
    EnrollUserTimeStepFunction(MyTimeStep);
  // AMR setup:
   //   if (adaptive==true)
    //      EnrollUserRefinementCondition(RefinementCondition);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
 //   Real x0 = 12.0/1024.0, y0 = 0.0, z0 = 0.0, r = 6.0/1024.0;
 
    Real G = four_pi_G / (4.0 * PI);
      
    Real dencurrent = 0., pcurrent = 0.;
  
    std::int64_t iseed = -1 - gid;
        
    Real alpha = rscale;
    
    Real rhostar = mscale / (4. * PI * std::sqrt(3.) * std::pow(alpha, 3.));
    Real pstar = four_pi_G / 6. * SQR(alpha * rhostar);
    
    if (!NON_BAROTROPIC_EOS){
        rhostar = 3.*SQR(cs/rscale) / (2.*PI * G);
        pstar = rhostar * SQR(cs);
    }
    
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "pstar = " << pstar << std::endl;
    std::cout << "isostar(0) = " << isostar(0.0)  << std::endl;
    std::cout << "Mach = " << vx / std::sqrt(pstar/rhostar * gam) << std::endl ;
    
    Real minstar = 0., mtot = 0.;
    
    for (int k = ks-1; k <= ke+1; ++k) {
        for (int j = js-1; j <= je+1; ++j) {
            for (int i = is-1; i <= ie+1; ++i) {
                Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k);
                Real dx = pcoord->dx1v(i), dy = pcoord->dx2v(j), dz = pcoord->dx3v(k);
                Real rsq = SQR(x-xstar)+SQR(y-ystar)+SQR(z-zstar);
                Real r2 = std::sqrt(rsq); //, rsq0 = 3.0;
                // Real r2corr = r2 / (1.+rsq0 / r2);
                Real theta = 1./std::sqrt(1.+rsq/SQR(alpha)/3.);
                
                dencurrent = rhostar * std::pow(theta, 5.);
                pcurrent = pstar * std::pow(theta, 6.);
                
                /*
                if (!NON_BAROTROPIC_EOS){
                    dencurrent = rhostar * isostar(r2/rscale);
                    pcurrent = dencurrent * SQR(cs);
                }
                */
                
                phydro->w(IDN,k,j,i) = std::max(dencurrent, bgdrho);
                // pressure scales with M^2, hence we need to downscale it twice
                phydro->w(IPR,k,j,i) = std::max(pcurrent, bgdpress);
                
                mtot += dencurrent * dx * dy * dz ;
                phydro->w(IM1,k,j,i) = vx;
                phydro->w(IM2,k,j,i) = vy;
                phydro->w(IM3,k,j,i) = vz;
                constexpr int scalar_norm = NSCALARS > 0 ? NSCALARS : 1.0;
                if(NSCALARS>0){
                    // entropy P/\rho^gamma
                    pscalars->s(0,k,j,i) = 1.0/scalar_norm * phydro->w(IPR,k,j,i) / std::pow(phydro->w(IDN,k,j,i), gam-1.);
                }
            }
        }
    }
    
    
    std::cout << "Mtot = " << mtot << "(" << minstar << ")\n";
    
    AthenaArray<Real> bb;
    bb.NewAthenaArray(3, ke+1, je+1, ie+1);
    peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);

}

Real faddfun(Real x){
    
    return geff; // 0.1 * std::sin(x * PI /30.);
    
}

// additional external gravity term:
void grav(MeshBlock *pmb, const Real time, const Real dt,
            const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
            const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
          AthenaArray<Real> &cons_scalar){
    // Real fadd = 1.0;
    // pmb->phydro->flux
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
        Real z = pmb->pcoord->x3v(k);
        for (int j=pmb->js; j<=pmb->je; ++j) {
            Real y = pmb->pcoord->x2v(j);
            for (int i=pmb->is; i<=pmb->ie; ++i) {
                Real x = pmb->pcoord->x1v(i), dx = pmb->pcoord->x1v(i);
                Real den = prim(IDN,k,j,i);
                cons(IM1,k,j,i) -=  faddfun(x) * dt * den  ;
                //        cons(IM2,k,j,i) -=  dt * den * fadd * y;
                //      cons(IM3,k,j,i) -=  dt * den * fadd * z;
                if (NON_BAROTROPIC_EOS){
                    cons(IEN,k,j,i) -= faddfun(x) * dt * prim(IM1,k,j,i) * den;
                    //Real dc =  0.5 * dt * (faddfun(x-dx/2.) * pmb->phydro->flux[X1DIR](IDN, k,j,i)+
                     //                     faddfun(x+dx/2.) * pmb->phydro->flux[X1DIR](IDN, k,j,i+1));
                    // Real press = prim(IEN,k,j,i);
                    // cons(IEN,k,j,i) -= dc;
                    /*
                    if (NSCALARS>0){
                        Real dp = prim(IPR,k,j,i)-(std::pow(prim(IDN,k,j,i),gam) * pmb->pscalars->r(0,k,j,i));
                       // if (dp < (-0.1 * prim(IPR,k,j,i)))cons(IEN,k,j,i) -= dp / (gam-1.) * 0.1;
                    }
 */
                }
            }
        }
    }
}

void psmear(MeshBlock *pmb, const Real time, const Real dt,
            const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
            const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
            AthenaArray<Real> &cons_scalar){
    
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
        Real z = pmb->pcoord->x3v(k);
        for (int j=pmb->js; j<=pmb->je; ++j) {
            Real y = pmb->pcoord->x2v(j);
            for (int i=pmb->is; i<=pmb->ie; ++i) {
                Real dp = prim(IPR,k,j,i)-(std::pow(prim(IDN,k,j,i),gam) * pmb->pscalars->r(0,k,j,i));
                if (dp < 0.)cons(IEN,k,j,i) -= dp / (gam-1.);
            }
        }
    }
}

Real MyTimeStep(MeshBlock *pmb)
{
    Real min_dt=FLT_MAX;
    bool worklimit = true;

    AthenaArray<Real> &w = pmb->phydro->w;
    AthenaArray<Real> &R = pmb->pscalars->r;

 for (int k=pmb->ks; k<=pmb->ke; ++k) {
   for (int j=pmb->js; j<=pmb->je; ++j) {
     for (int i=pmb->is; i<=pmb->ie; ++i) {
       Real r0 = R(0, k, j, i);
       Real x = pmb->pcoord->x1v(i), y = pmb->pcoord->x2v(j), z = pmb->pcoord->x3v(k);
       Real dx = pmb->pcoord->dx1v(i), dy = pmb->pcoord->dx2v(j), dz = pmb->pcoord->dx3v(k);

       Real dr = std::min(dx, std::min(dy, dz));
       Real r1sq = SQR(x)+SQR(y)+SQR(z); // distance to the BH
       Real csq = gam * w(IPR,k,j,i) / w(IDN,k,j,i);
       Real vabs = std::sqrt(std::max(SQR(w(IM1,k,j,i)) + SQR(w(IM2,k,j,i)) + SQR(w(IM3,k,j,i)), csq));
       Real dtP = csq / vabs  / geff * gstepnumber;
       // dtR = dr * () / std::sqrt(SQR(w(IM1, k, j, i))+SQR(w(IM2, k, j, i))+SQR(w(IM3, k, j, i))+\
w(IEN, k, j, i)/w(IDN, k, j, i));
       if (r0 > refR)min_dt = std::min(min_dt, dtP); // std::min(dtR, dtG));
     }
   }
 }
 return min_dt;
}

namespace{
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
