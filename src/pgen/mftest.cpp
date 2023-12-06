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

namespace{
  Real four_pi_G, gam;
  Real rho, temp;
  Real rcore, amp; // xcore = rstarcore/rstar
  int loop_xc, loop_yc, loop_rad; // multipolar structure in magnetic field or convection velocity field
  Real vx, vy, vz; // transverse velocity
Real threshold;
}

int RefinementCondition(MeshBlock *pmb);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  gam = pin->GetReal("hydro","gamma");

    loop_xc = pin->GetReal("problem","xc");
    loop_yc = pin->GetReal("problem","yc");
    loop_rad = pin->GetReal("problem","rloop");

    amp = pin->GetReal("problem","amp");
    
    rho = pin->GetReal("problem","rho");
    
    temp = pin->GetReal("problem","temp");

    vx = pin->GetReal("problem","vx");
    vy = pin->GetReal("problem","vy");
    vz = pin->GetReal("problem","vz");
    
    // SetFourPiG(four_pi_G);

    // AMR setup:
    if (adaptive==true)
    {
        EnrollUserRefinementCondition(RefinementCondition);
        threshold = pin->GetReal("problem","thr");
    }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for gravity from a binary
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
 //   Real x0 = 12.0/1024.0, y0 = 0.0, z0 = 0.0, r = 6.0/1024.0;
 
  Real G = four_pi_G / (4.0 * PI);
  
  Real By = amp;
    
  Real dencurrent = 0., pcurrent = 0.;
      
    AthenaArray<Real> ax, ay, az;

    int nx1 = block_size.nx1 + 2*NGHOST;
    int nx2 = block_size.nx2 + 2*NGHOST;
    int nx3 = block_size.nx3 + 2*NGHOST;
    ax.NewAthenaArray(nx3, nx2, nx1);
    ay.NewAthenaArray(nx3, nx2, nx1);
    az.NewAthenaArray(nx3, nx2, nx1);

    std::cout << nx1 << " " << nx2 << " " << nx3 << "\n";
    std::cout << ks << " " << ke << "\n";
    std::cout << js << " " << je << "\n";
    std::cout << is << " " << ie << "\n";

    Real is1 = is-NGHOST, ie1 = ie+NGHOST, js1 = js-NGHOST, je1 = je+NGHOST, ks1 = ks-NGHOST, ke1 = ke+NGHOST;

  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
          // std::cout << i << " " << j << " " << k << "\n";
          Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k);
          Real xf = pcoord->x1f(i), yf = pcoord->x2f(j), zf = pcoord->x3f(k);
          Real dx = pcoord->dx1v(i), dy = pcoord->dx2v(j), dz = pcoord->dx3v(k);

              phydro->w(IDN,k,j,i) = rho;
              phydro->w(IPR,k,j,i) = rho * temp;
              
              // phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i);
              // velocities:
              phydro->w(IM1,k,j,i) = vx;
              phydro->w(IM2,k,j,i) = vy;
              phydro->w(IM3,k,j,i) = vz;
  
          // pfield->b.x1f(k,j,i) = 0.;
          
          ax(k,j,i) = 0.0;
          ay(k,j,i) = 0.0;
          if ((SQR(pcoord->x1f(i)-loop_xc) + SQR(pcoord->x2f(j)-loop_yc)) < SQR(loop_rad)) {
            az(k,j,i) = amp*std::max(loop_rad - std::sqrt(SQR(xf-loop_xc) + SQR(yf-loop_yc)), 0.);
            az(k,j+1,i) = amp*std::max(loop_rad - std::sqrt(SQR(xf-loop_xc) +
                                               SQR(yf+dy-loop_yc)), 0.);
            az(k,j-1,i) = amp*std::max(loop_rad - std::sqrt(SQR(xf-loop_xc) +
                                                 SQR(yf-dy-loop_yc)),0.);
              az(k,j,i-1) = amp*std::max(loop_rad - std::sqrt(SQR(xf-dx-loop_xc) +
                                                   SQR(yf-loop_yc)),0.);
              az(k,j,i+1) = amp*std::max(loop_rad - std::sqrt(SQR(xf+dx-loop_xc) +
                                                   SQR(yf-loop_yc)),0.);
              az(k,j+1,i+1) = amp*std::max(loop_rad - std::sqrt(SQR(xf+dx-loop_xc) +
                                                   SQR(yf+dy-loop_yc)),0.);
              az(k,j-1,i+1) = amp*std::max(loop_rad - std::sqrt(SQR(xf+dx-loop_xc) +
                                                   SQR(yf-dy-loop_yc)),0.);
              az(k,j+1,i-1) = amp*std::max(loop_rad - std::sqrt(SQR(xf-dx-loop_xc) +
                                                   SQR(yf+dy-loop_yc)),0.);
              az(k,j-1,i-1) = amp*std::max(loop_rad - std::sqrt(SQR(xf-dx-loop_xc) +
                                                   SQR(yf-dy-loop_yc)),0.);

            if(NSCALARS>0)pscalars->s(0,k,j,i) = rho;

          } else {
            az(k,j,i) = 0.0;

                  if(NSCALARS>0)pscalars->s(0,k,j,i) = 0.;

          }

          /*
        if ((x> xmin) && (x<xmax) && (y> ymin) && (y<ymax)){
            pfield->b.x2f(k,j,i) = By;
            pfield->b.x2f(k,j+1,i) = By;
            if(NSCALARS>0)pscalars->s(0,k,j,i) = rho;
       }
           */
        // pfield->b.x3f(k,j,i) = 0.;
         
      }
    }
  }
    
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          pfield->b.x1f(k,j,i) = (az(k,j+1,i) - az(k,j,i))/pcoord->dx2f(j) -
                                 (ay(k+1,j,i) - ay(k,j,i))/pcoord->dx3f(k);
        }
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x2f(k,j,i) = (ax(k+1,j,i) - ax(k,j,i))/pcoord->dx3f(k) -
                                 (az(k,j,i+1) - az(k,j,i))/pcoord->dx1f(i);
        }
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is1; i<=ie1; i++) {
          pfield->b.x3f(k,j,i) = (ay(k,j,i+1) - ay(k,j,i))/pcoord->dx1f(i) -
                                 (ax(k,j+1,i) - ax(k,j,i))/pcoord->dx2f(j);
        }
      }
    }
    
  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is1, ie1, js1, je1, ks1, ke1); //
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie, js, je, ks, ke);

}

int RefinementCondition(MeshBlock *pmb) {
  int f2 = pmb->pmy_mesh->f2, f3 = pmb->pmy_mesh->f3;
  AthenaArray<Real> &r = pmb->pscalars->r;
  Real maxeps = 0.0;
  if (f3) {
    for (int n=0; n<NSCALARS; ++n) {
      for (int k=pmb->ks-1; k<=pmb->ke+1; k++) {
        for (int j=pmb->js-1; j<=pmb->je+1; j++) {
          for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
            Real eps = std::sqrt(SQR(0.5*(r(n,k,j,i+1) + r(n,k,j,i-1))-r(n,k,j,i))
                                 + SQR(0.5*(r(n,k,j+1,i) + r(n,k,j-1,i))-r(n,k,j,i))
                                 + SQR(0.5*(r(n,k+1,j,i) + r(n,k-1,j,i))-r(n,k,j,i)));
            // /r(n,k,j,i); Do not normalize by scalar, since (unlike IDN and IPR) there
            // are are no physical floors / r=0 might be allowed. Compare w/ blast.cpp.
            maxeps = std::max(maxeps, eps);
          }
        }
      }
    }
  } else if (f2) {
    int k = pmb->ks;
    for (int n=0; n<NSCALARS; ++n) {
      for (int j=pmb->js-1; j<=pmb->je+1; j++) {
        for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
 //         Real eps = std::sqrt(SQR(0.5*(r(n,k,j,i+1) + r(n,k,j,i-1))-r(n,k,j,i))
  //                             + SQR(0.5*(r(n,k,j+1,i) + r(n,k,j-1,i))-r(n,k,j,i))); // /r(n,k,j,i);
            Real eps = std::sqrt(SQR(0.5*(r(n,k,j,i+1) - r(n,k,j,i-1))-r(n,k,j,i)*0.)
                                 + SQR(0.5*(r(n,k,j+1,i) - r(n,k,j-1,i))-r(n,k,j,i)*0.)); // /r(n,k,j,i);

          maxeps = std::max(maxeps, eps);
        }
      }
    }
  } else {
    return 0;
  }

  if (maxeps > threshold) return 1;
  if (maxeps < 0.25*threshold) return -1;
  return 0;
}
