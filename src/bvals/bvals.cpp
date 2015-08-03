
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================

// Primary header
#include "bvals.hpp"

// C++ headers
#include <iostream>   // endl
#include <iomanip>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cstring>    // memcpy
#include <cstdlib>

// Athena headers
#include "../athena.hpp"          // Real
#include "../athena_arrays.hpp"   // AthenaArray
#include "../mesh.hpp"            // MeshBlock
#include "../fluid/fluid.hpp"     // Fluid
#include "../coordinates/coordinates.hpp" // Coordinates
#include "../parameter_input.hpp" // ParameterInput

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

static NeighborIndexes ni_[56];
static int bufid_[56];

//======================================================================================
//! \file bvals.cpp
//  \brief implements functions that initialize/apply BCs on each dir
//======================================================================================

// BoundaryValues constructor - sets functions for the appropriate
// boundary conditions at each of the 6 dirs of a MeshBlock

BoundaryValues::BoundaryValues(MeshBlock *pmb, ParameterInput *pin)
{
  pmy_mblock_ = pmb;

// Set BC functions for each of the 6 boundaries in turn -------------------------------
// Inner x1
  nface_=2; nedge_=0;
  switch(pmb->block_bcs[inner_x1]){
    case 1:
      FluidBoundary_[inner_x1] = ReflectInnerX1;
      FieldBoundary_[inner_x1] = ReflectInnerX1;
      break;
    case 2:
      FluidBoundary_[inner_x1] = OutflowInnerX1;
      FieldBoundary_[inner_x1] = OutflowInnerX1;
      break;
    case -1: // block boundary
    case 3: // do nothing, useful for user-enrolled BCs
    case 4: // periodic boundary
      FluidBoundary_[inner_x1] = NULL;
      FieldBoundary_[inner_x1] = NULL;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
          << "Flag ix1_bc=" << pmb->block_bcs[inner_x1] << " not valid" << std::endl;
      throw std::runtime_error(msg.str().c_str());
   }

// Outer x1
  switch(pmb->block_bcs[outer_x1]){
    case 1:
      FluidBoundary_[outer_x1] = ReflectOuterX1;
      FieldBoundary_[outer_x1] = ReflectOuterX1;
      break;
    case 2:
      FluidBoundary_[outer_x1] = OutflowOuterX1;
      FieldBoundary_[outer_x1] = OutflowOuterX1;
      break;
    case -1: // block boundary
    case 3: // do nothing, useful for user-enrolled BCs
    case 4: // periodic boundary
      FluidBoundary_[outer_x1] = NULL;
      FieldBoundary_[outer_x1] = NULL;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
          << "Flag ox1_bc=" << pmb->block_bcs[outer_x1] << " not valid" << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }

  if (pmb->block_size.nx2 > 1) {
    nface_=4; nedge_=4;
// Inner x2
    switch(pmb->block_bcs[inner_x2]){
      case 1:
        FluidBoundary_[inner_x2] = ReflectInnerX2;
        FieldBoundary_[inner_x2] = ReflectInnerX2;
        break;
      case 2:
        FluidBoundary_[inner_x2] = OutflowInnerX2;
        FieldBoundary_[inner_x2] = OutflowInnerX2;
        break;
      case -1: // block boundary
      case 3: // do nothing, useful for user-enrolled BCs
      case 4: // periodic boundary
        FluidBoundary_[inner_x2] = NULL;
        FieldBoundary_[inner_x2] = NULL;
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
            << "Flag ix2_bc=" << pmb->block_bcs[inner_x2] << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
     }

// Outer x2
    switch(pmb->block_bcs[outer_x2]){
      case 1:
        FluidBoundary_[outer_x2] = ReflectOuterX2;
        FieldBoundary_[outer_x2] = ReflectOuterX2;
        break;
      case 2:
        FluidBoundary_[outer_x2] = OutflowOuterX2;
        FieldBoundary_[outer_x2] = OutflowOuterX2;
        break;
      case -1: // block boundary
      case 3: // do nothing, useful for user-enrolled BCs
      case 4: // periodic boundary
        FluidBoundary_[outer_x2] = NULL;
        FieldBoundary_[outer_x2] = NULL;
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
            << "Flag ox2_bc=" << pmb->block_bcs[outer_x2] << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
    }
  }

  if (pmb->block_size.nx3 > 1) {
    nface_=6; nedge_=12;
// Inner x3
    switch(pmb->block_bcs[inner_x3]){
      case 1:
        FluidBoundary_[inner_x3] = ReflectInnerX3;
        FieldBoundary_[inner_x3] = ReflectInnerX3;
        break;
      case 2:
        FluidBoundary_[inner_x3] = OutflowInnerX3;
        FieldBoundary_[inner_x3] = OutflowInnerX3;
        break;
      case -1: // block boundary
      case 3: // do nothing, useful for user-enrolled BCs
      case 4: // periodic boundary
        FluidBoundary_[inner_x3] = NULL;
        FieldBoundary_[inner_x3] = NULL;
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
            << "Flag ix3_bc=" << pmb->block_bcs[inner_x3] << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
     }

// Outer x3
    switch(pmb->block_bcs[outer_x3]){
      case 1:
        FluidBoundary_[outer_x3] = ReflectOuterX3;
        FieldBoundary_[outer_x3] = ReflectOuterX3;
        break;
      case 2:
        FluidBoundary_[outer_x3] = OutflowOuterX3;
        FieldBoundary_[outer_x3] = OutflowOuterX3;
        break;
      case -1: // block boundary
      case 3: // do nothing, useful for user-enrolled BCs
      case 4: // periodic boundary
        FluidBoundary_[outer_x3] = NULL;
        FieldBoundary_[outer_x3] = NULL;
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues constructor" << std::endl
            << "Flag ox3_bc=" << pmb->block_bcs[outer_x3] << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
    }
  }

  // Clear flags and requests
  for(int l=0;l<NSTEP;l++) {
    for(int i=0;i<56;i++){
      fluid_flag_[l][i]=boundary_waiting;
      field_flag_[l][i]=boundary_waiting;
      fluid_send_[l][i]=NULL;
      fluid_recv_[l][i]=NULL;
      field_send_[l][i]=NULL;
      field_recv_[l][i]=NULL;
#ifdef MPI_PARALLEL
      req_fluid_send_[l][i]=MPI_REQUEST_NULL;
      req_fluid_recv_[l][i]=MPI_REQUEST_NULL;
#endif
    }
    for(int i=0;i<6;i++){
      flcor_send_[l][i]=NULL;
#ifdef MPI_PARALLEL
      req_flcor_send_[l][i]=MPI_REQUEST_NULL;
#endif
      for(int j=0;j<=1;j++) {
        for(int k=0;k<=1;k++) {
          flcor_recv_[l][i][j][k]=NULL;
#ifdef MPI_PARALLEL
          req_flcor_recv_[l][i][j][k]=MPI_REQUEST_NULL;
#endif
        }
      }
    }
  }
  // Allocate Buffers
  for(int l=0;l<NSTEP;l++) {
    for(int n=0;n<pmb->pmy_mesh->maxneighbor_;n++) {
      int size=((ni_[n].ox1==0)?pmb->block_size.nx1:NGHOST)
              *((ni_[n].ox2==0)?pmb->block_size.nx2:NGHOST)
              *((ni_[n].ox3==0)?pmb->block_size.nx3:NGHOST);
      if(pmb->pmy_mesh->multilevel==true) {
        int cng=pmb->cnghost, cng1=0, cng2=0, cng3=0;
        if(pmb->block_size.nx2>1) cng1=cng, cng2=cng;
        if(pmb->block_size.nx3>1) cng3=cng;
        int f2c=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2):NGHOST)
               *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2):NGHOST)
               *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2):NGHOST);
        int c2f=((ni_[n].ox1==0)?((pmb->block_size.nx1+1)/2+cng1):cng)
               *((ni_[n].ox2==0)?((pmb->block_size.nx2+1)/2+cng2):cng)
               *((ni_[n].ox3==0)?((pmb->block_size.nx3+1)/2+cng3):cng);
        size=std::max(size,c2f);
        size=std::max(size,f2c);
      }
      size*=NFLUID;
      fluid_send_[l][n]=new Real[size];
      fluid_recv_[l][n]=new Real[size];
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for(int l=0;l<NSTEP;l++) {
      for(int n=0;n<pmb->pmy_mesh->maxneighbor_;n++) {
        int size1=((ni_[n].ox1==0)?(pmb->block_size.nx1+1):NGHOST)
                 *((ni_[n].ox2==0)?(pmb->block_size.nx2):NGHOST)
                 *((ni_[n].ox3==0)?(pmb->block_size.nx3):NGHOST);
        int size2=((ni_[n].ox1==0)?(pmb->block_size.nx1):NGHOST)
                 *((ni_[n].ox2==0)?(pmb->block_size.nx2+1):NGHOST)
                 *((ni_[n].ox3==0)?(pmb->block_size.nx3):NGHOST);
        int size3=((ni_[n].ox1==0)?(pmb->block_size.nx1):NGHOST)
                 *((ni_[n].ox2==0)?(pmb->block_size.nx2):NGHOST)
                 *((ni_[n].ox3==0)?(pmb->block_size.nx3+1):NGHOST);
        if(pmb->pmy_mesh->multilevel==true) {
          // *** need to implement
        }
        int size=size1+size2+size3;
        field_send_[l][n]=new Real[size];
        field_recv_[l][n]=new Real[size];
      }
    }
  }

  if(pmb->pmy_mesh->multilevel==true) { // SMR or AMR
    // allocate arrays for volumes in the finer level
    int nc1=pmb->block_size.nx1+2*NGHOST;
    int nc2=pmb->block_size.nx2+2*NGHOST;
    int nc3=pmb->block_size.nx3+2*NGHOST;
    fvol_[0][0].NewAthenaArray(nc1+1);
    fvol_[0][1].NewAthenaArray(nc1+1);
    fvol_[1][0].NewAthenaArray(nc1+1);
    fvol_[1][1].NewAthenaArray(nc1+1);
    sarea_[0].NewAthenaArray(nc1);
    sarea_[1].NewAthenaArray(nc1);
    int size[6], im, jm, km;
    // allocate flux correction buffer
    size[0]=size[1]=(pmb->block_size.nx2+1)/2*(pmb->block_size.nx3+1)/2*NFLUID;
    size[2]=size[3]=(pmb->block_size.nx1+1)/2*(pmb->block_size.nx3+1)/2*NFLUID;
    size[4]=size[5]=(pmb->block_size.nx1+1)/2*(pmb->block_size.nx2+1)/2*NFLUID;
    if(pmb->block_size.nx3>1) { // 3D
      jm=2, km=2;
      surface_flux_[inner_x1].NewAthenaArray(NFLUID, nc3, nc2);
      surface_flux_[outer_x1].NewAthenaArray(NFLUID, nc3, nc2);
      surface_flux_[inner_x2].NewAthenaArray(NFLUID, nc3, nc1);
      surface_flux_[outer_x2].NewAthenaArray(NFLUID, nc3, nc1);
      surface_flux_[inner_x3].NewAthenaArray(NFLUID, nc2, nc1);
      surface_flux_[outer_x3].NewAthenaArray(NFLUID, nc2, nc1);
    }
    else if(pmb->block_size.nx2>1) { // 2D
      jm=1, km=2;
      surface_flux_[inner_x1].NewAthenaArray(NFLUID, 1, nc2);
      surface_flux_[outer_x1].NewAthenaArray(NFLUID, 1, nc2);
      surface_flux_[inner_x2].NewAthenaArray(NFLUID, 1, nc1);
      surface_flux_[outer_x2].NewAthenaArray(NFLUID, 1, nc1);
    }
    else { // 1D
      jm=1, km=1;
      surface_flux_[inner_x1].NewAthenaArray(NFLUID, 1, 1);
      surface_flux_[outer_x1].NewAthenaArray(NFLUID, 1, 1);
    }
    for(int l=0;l<NSTEP;l++) {
      for(int i=0;i<nface_;i++){
        flcor_send_[l][i]=new Real[size[i]];
        for(int j=0;j<jm;j++) {
          for(int k=0;k<km;k++)
            flcor_recv_[l][i][j][k]=new Real[size[i]];
        }
      }
    }
    // allocate prolongation buffer
    int ncc1=pmb->block_size.nx1/2+2*pmb->cnghost;
    int ncc2=1;
    if(pmb->block_size.nx2>1) ncc2=pmb->block_size.nx2/2+2*pmb->cnghost;
    int ncc3=1;
    if(pmb->block_size.nx3>1) ncc3=pmb->block_size.nx3/2+2*pmb->cnghost;
    coarse_cons_.NewAthenaArray(NFLUID,ncc3,ncc2,ncc1);

    if (MAGNETIC_FIELDS_ENABLED) {
      coarse_b_.x1f.NewAthenaArray(ncc3,ncc2,ncc1+1);
      coarse_b_.x2f.NewAthenaArray(ncc3,ncc2+1,ncc1);
      coarse_b_.x3f.NewAthenaArray(ncc3+1,ncc2,ncc1);
      int fsize[6], esize[12];
      // allocate EMF correction buffer
      if(pmb->block_size.nx3>1) { // 3D
        fsize[0]=fsize[1]=(pmb->block_size.nx2/2+1)*(pmb->block_size.nx3/2)
                         +(pmb->block_size.nx2/2)*(pmb->block_size.nx3/2+1);
        fsize[2]=fsize[3]=(pmb->block_size.nx1/2+1)*(pmb->block_size.nx3/2)
                         +(pmb->block_size.nx1/2)*(pmb->block_size.nx3/2+1);
        fsize[4]=fsize[5]=(pmb->block_size.nx1/2+1)*(pmb->block_size.nx2/2)
                         +(pmb->block_size.nx1/2)*(pmb->block_size.nx2/2+1);
        esize[0]=esize[1]=esize[2]=esize[3]=pmb->block_size.nx3/2;
        esize[4]=esize[5]=esize[6]=esize[7]=pmb->block_size.nx2/2;
        esize[8]=esize[9]=esize[10]=esize[11]=pmb->block_size.nx1/2;
      }
      else if(pmb->block_size.nx2>1) { // 2D
        fsize[0]=fsize[1]=(pmb->block_size.nx2/2+1)+pmb->block_size.nx2/2;
        fsize[2]=fsize[3]=(pmb->block_size.nx1/2+1)+pmb->block_size.nx1/2;
        for(int i=0; i<nedge_; i++)
          esize[i]=1;
      }
      else { // 1D
        fsize[0]=fsize[1]=2;
      }
      for(int l=0;l<NSTEP;l++) {
        for(int i=0;i<nface_;i++){
          emfcor_send_[l][i]=new Real[fsize[i]];
          for(int j=0;j<jm;j++) {
            for(int k=0;k<km;k++)
              emfcor_recv_[l][i][j][k]=new Real[fsize[i]];
          }
        }
        for(int i=0;i<nedge_;i++){
          emfcor_send_[l][nface_+i]=new Real[esize[i]];
          for(int j=0;j<jm;j++)
            emfcor_recv_[l][nface_+i][0][j]=new Real[esize[i]];
        }
      }
    }
  }
}

// destructor

BoundaryValues::~BoundaryValues()
{
  MeshBlock *pmb=pmy_mblock_;
  for(int l=0;l<NSTEP;l++) {
    for(int i=0;i<pmb->pmy_mesh->maxneighbor_;i++) {
      delete [] fluid_send_[l][i];
      delete [] fluid_recv_[l][i];
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for(int l=0;l<NSTEP;l++) {
      for(int i=0;i<pmb->pmy_mesh->maxneighbor_;i++) { 
        delete [] field_send_[l][i];
        delete [] field_recv_[l][i];
      }
    }
  }
  if(pmb->pmy_mesh->multilevel==true) {
    fvol_[0][0].DeleteAthenaArray();
    fvol_[0][1].DeleteAthenaArray();
    fvol_[1][0].DeleteAthenaArray();
    fvol_[1][1].DeleteAthenaArray();
    sarea_[0].DeleteAthenaArray();
    sarea_[1].DeleteAthenaArray();
    for(int r=0;r<nface_;r++)
      surface_flux_[r].DeleteAthenaArray();
    for(int l=0;l<NSTEP;l++) {
      for(int i=0;i<nface_;i++){
        delete [] flcor_send_[l][i];
        for(int j=0;j<2;j++) {
          for(int k=0;k<2;k++)
            delete [] flcor_recv_[l][i][j][k];
        }
      }
    }
    coarse_cons_.DeleteAthenaArray();
//  coarse_prim_.DeleteAthenaArray();
    if (MAGNETIC_FIELDS_ENABLED) {
      coarse_b_.x1f.DeleteAthenaArray();
      coarse_b_.x2f.DeleteAthenaArray();
      coarse_b_.x3f.DeleteAthenaArray();
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::Initialize(void)
//  \brief Initialize MPI requests
void BoundaryValues::Initialize(void)
{
#ifdef MPI_PARALLEL
  MeshBlock* pmb=pmy_mblock_;
  long int lx1, lx2, lx3;
  int mylevel,myox1, myox2, myox3;
  int tag;
  int cng1, cng2, cng3;
  int ssize, rsize;
  cng1=pmb->cnghost;
  cng2=(pmb->block_size.nx2>1)?cng1:0;
  cng3=(pmb->block_size.nx3>1)?cng1:0;
  pmb->uid.GetLocation(lx1,lx2,lx3,mylevel);
  myox1=((int)(lx1&1L));
  myox2=((int)(lx2&1L));
  myox3=((int)(lx3&1L));

  // check edge neighbors
  if(pmb->pmy_mesh->multilevel==true) {
    for(int n=0;n<pmb->nneighbor;n++) {
      NeighborBlock& nb=pmb->neighbor[n];
      if(nb.type==neighbor_edge) {
        int nlev=std::max(nb.level,mylevel);
        edge_flag_[ne.eid]=true;
        if(ne.eid>=0 && ne.eid<4) {
          if(pmb->nblevel[1][nb.ox2][1]==nlev
          || pmb->nblevel[nb.ox1][1][1]==nlev) edge_flag_[ne.eid]=false; // already corrected by face
        }
        else if(ne.eid>=4 && ne.eid<8) {
          if(pmb->nblevel[1][1][nb.ox3]==nlev
          || pmb->nblevel[nb.ox1][1][1]==nlev) edge_flag_[ne.eid]=false; // already corrected by face
        }
        else if(ne.eid>=8 && ne.eid<12) {
          if(pmb->nblevel[1][1][nb.ox3]==nlev
          || pmb->nblevel[1][nb.ox2][1]==nlev) edge_flag_[ne.eid]=false; // already corrected by face
        }
      }
    }
  }

  for(int l=0;l<NSTEP;l++) {
    for(int n=0;n<pmb->nneighbor;n++) {
      NeighborBlock& nb=pmb->neighbor[n];
      if(nb.rank!=myrank) {
        if(nb.level==mylevel) { // same
          ssize=rsize=((nb.ox1==0)?pmb->block_size.nx1:NGHOST)
                     *((nb.ox2==0)?pmb->block_size.nx2:NGHOST)
                     *((nb.ox3==0)?pmb->block_size.nx3:NGHOST);
        }
        else if(nb.level<mylevel) { // coarser
          ssize=((nb.ox1==0)?((pmb->block_size.nx1+1)/2):NGHOST)
               *((nb.ox2==0)?((pmb->block_size.nx2+1)/2):NGHOST)
               *((nb.ox3==0)?((pmb->block_size.nx3+1)/2):NGHOST);
          rsize=((nb.ox1==0)?((pmb->block_size.nx1+1)/2+cng1):cng1)
               *((nb.ox2==0)?((pmb->block_size.nx2+1)/2+cng2):cng2)
               *((nb.ox3==0)?((pmb->block_size.nx3+1)/2+cng3):cng3);
        }
        else { // finer
          ssize=((nb.ox1==0)?((pmb->block_size.nx1+1)/2+cng1):cng1)
               *((nb.ox2==0)?((pmb->block_size.nx2+1)/2+cng2):cng2)
               *((nb.ox3==0)?((pmb->block_size.nx3+1)/2+cng3):cng3);
          rsize=((nb.ox1==0)?((pmb->block_size.nx1+1)/2):NGHOST)
               *((nb.ox2==0)?((pmb->block_size.nx2+1)/2):NGHOST)
               *((nb.ox3==0)?((pmb->block_size.nx3+1)/2):NGHOST);
        }
        ssize*=NFLUID; rsize*=NFLUID;
        // specify the offsets in the view point of the target block: flip ox? signs
        tag=CreateMPITag(nb.lid, l, tag_fluid, nb.targetid);
        MPI_Send_init(fluid_send_[l][nb.bufid],ssize,MPI_ATHENA_REAL,
                      nb.rank,tag,MPI_COMM_WORLD,&req_fluid_send_[l][nb.bufid]);
        tag=CreateMPITag(pmb->lid, l, tag_fluid, nb.bufid);
        MPI_Recv_init(fluid_recv_[l][nb.bufid],rsize,MPI_ATHENA_REAL,
                      nb.rank,tag,MPI_COMM_WORLD,&req_fluid_recv_[l][nb.bufid]);

        // flux correction
        if(pmb->pmy_mesh->multilevel==true && nb.type==neighbor_face) {
          int fi1, fi2, size;
          if(nb.fid==0 || nb.fid==1)
            fi1=myox2, fi2=myox3, size=((pmb->block_size.nx2+1)/2)*((pmb->block_size.nx3+1)/2);
          else if(nb.fid==2 || nb.fid==3)
            fi1=myox1, fi2=myox3, size=((pmb->block_size.nx1+1)/2)*((pmb->block_size.nx3+1)/2);
          else if(nb.fid==4 || nb.fid==5)
            fi1=myox1, fi2=myox2, size=((pmb->block_size.nx1+1)/2)*((pmb->block_size.nx2+1)/2);
          size*=NFLUID;
          if(nb.level<mylevel) { // send to coarser
            tag=CreateMPITag(nb.lid, l, tag_flcor, ((nb.fid^1)<<2)|(fi2<<1)|fi1);
            MPI_Send_init(flcor_send_[l][nb.fid],size,MPI_ATHENA_REAL,
                nb.rank,tag,MPI_COMM_WORLD,&req_flcor_send_[l][nb.fid]);
          }
          else if(nb.level>mylevel) { // receive from finer
            tag=CreateMPITag(pmb->lid, l, tag_flcor, (nb.fid<<2)|(nb.fi2<<1)|nb.fi1);
            MPI_Recv_init(flcor_recv_[l][nb.fid][nb.fi2][nb.fi1],size,MPI_ATHENA_REAL,
                nb.rank,tag,MPI_COMM_WORLD,&req_flcor_recv_[l][nb.fid][nb.fi2][nb.fi1]);
          }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
          int size1, size2, size3;
          if(pmb->pmy_mesh->multilevel==false) { // uniform
            size1=((nb.ox1==0)?(pmb->block_size.nx1+1):NGHOST)
                 *((nb.ox2==0)?(pmb->block_size.nx2):NGHOST)
                 *((nb.ox3==0)?(pmb->block_size.nx3):NGHOST);
            size2=((nb.ox1==0)?(pmb->block_size.nx1):NGHOST)
                 *((nb.ox2==0)?(pmb->block_size.nx2+1):NGHOST)
                 *((nb.ox3==0)?(pmb->block_size.nx3):NGHOST);
            size3=((nb.ox1==0)?(pmb->block_size.nx1):NGHOST)
                 *((nb.ox2==0)?(pmb->block_size.nx2):NGHOST)
                 *((nb.ox3==0)?(pmb->block_size.nx3+1):NGHOST);
          }
          else {
            if(nb.level==mylevel) { // same
              // ****** need to implement ******
            }
            else if(nb.level<mylevel) { // coarser
              // ****** need to implement ******
            }
            else { // finer
              // ****** need to implement ******
            }
          }
          ssize=size1+size2+size3; rsize=ssize;
          // specify the offsets in the view point of the target block: flip ox? signs
          tag=CreateMPITag(nb.lid, l, tag_field, nb.targetid);
          MPI_Send_init(field_send_[l][nb.bufid],ssize,MPI_ATHENA_REAL,
                        nb.rank,tag,MPI_COMM_WORLD,&req_field_send_[l][nb.bufid]);
          tag=CreateMPITag(pmb->lid, l, tag_field, nb.bufid);
          MPI_Recv_init(field_recv_[l][nb.bufid],rsize,MPI_ATHENA_REAL,
                        nb.rank,tag,MPI_COMM_WORLD,&req_field_recv_[l][nb.bufid]);
          // EMF correction
          if(pmb->pmy_mesh->multilevel==true) {
            if(nb.type==neighbor_face) { // face
              int fi1, fi2, size;
              if(pmb->block_size.nx3 > 1) { // 3D
                if(nb.fid==inner_x1 || nb.fid==outer_x1) {
                  fi1=myox2; fi2=myox3;
                  size=(pmb->block_size.nx2/2+1)*(pmb->block_size.nx3/2)
                      +(pmb->block_size.nx2/2)*(pmb->block_size.nx3/2+1);
                }
                else if(nb.fid==inner_x2 || nb.fid==outer_x2) {
                  fi1=myox1; fi2=myox3;
                  size=(pmb->block_size.nx1/2+1)*(pmb->block_size.nx3/2)
                      +(pmb->block_size.nx1/2)*(pmb->block_size.nx3/2+1);
                }
                else if(nb.fid==inner_x3 || nb.fid==outer_x3) {
                  fi1=myox1; fi2=myox2;
                  size=(pmb->block_size.nx1/2+1)*(pmb->block_size.nx2/2)
                      +(pmb->block_size.nx1/2)*(pmb->block_size.nx2/2+1);
                }
              }
              else if(pmb->block_size.nx2 > 1) { // 2D
                if(nb.fid==inner_x1 || nb.fid==outer_x1) {
                  size=(pmb->block_size.nx2/2+1)+pmb->block_size.nx2/2;
                  fi1=myox2; fi2=0;
                }
                else if(nb.fid==inner_x2 || nb.fid==outer_x2) {
                  size=(pmb->block_size.nx1/2+1)+pmb->block_size.nx1/2;
                  fi1=myox1; fi2=0;
                }
              }
              else { // 1D
                size=2; fi1=0; fi2=0;
              }
              if(nb.level<mylevel) { // send to coarser
                tag=CreateMPITag(nb.lid, l, tag_emfcor_face, ((nb.fid^1)<<2)|(fi2<<1)|fi1);
                MPI_Send_init(emfcor_fsend_[l][nb.fid],size,MPI_ATHENA_REAL,
                    nb.rank,tag,MPI_COMM_WORLD,&req_emfcor_fsend_[l][nb.fid]);
              }
              else if(nb.level>mylevel) { // receive from finer
                tag=CreateMPITag(pmb->lid, l, tag_emfcor_face, (nb.fid<<2)|(nb.fi2<<1)|nb.fi1);
                MPI_Recv_init(emfcor_frecv_[l][nb.fid][nb.fi2][nb.fi1],size,MPI_ATHENA_REAL,
                    nb.rank,tag,MPI_COMM_WORLD,&req_emfcor_frecv_[l][nb.fid][nb.fi2][nb.fi1]);
              }
            }
            else if(nb.type==neighbor_edge) { // edge
              int fi1, size;
              if(edge_flag_[nb.eid]==false) continue;
              if(pmb->block_size.nx3 > 1) { // 3D
                if(nb.eid>=0 && nb.eid<4) {
                  size=pmb->block_size.nx3/2;
                  fi1=myox3;
                }
                else if(nb.eid>=4 && nb.eid<8) {
                  size=pmb->block_size.nx2/2;
                  fi1=myox2;
                }
                else if(nb.eid>=8 && nb.eid<12) {
                  size=pmb->block_size.nx1/2;
                  fi1=myox1;
                }
              }
              else if(pmb->block_size.nx2 > 1) { // 2D
                size=1; fi1=0;
              }
              if(nb.level<mylevel) { // send to coarser
                tag=CreateMPITag(nb.lid, l, tag_emfcor_edge, ((nb.eid^3)<<1)|fi1);
                MPI_Send_init(emfcor_esend_[l][nb.eid],size,MPI_ATHENA_REAL,
                    nb.rank,tag,MPI_COMM_WORLD,&req_emfcor_esend_[l][nb.eid]);
              }
              else if(nb.level>mylevel) { // receive from finer
                tag=CreateMPITag(pmb->lid, l, tag_emfcor_edge, (nb.eid<<1)|nb.fi1);
                MPI_Recv_init(emfcor_erecv_[l][nb.eid][nb.fi1],size,MPI_ATHENA_REAL,
                    nb.rank,tag,MPI_COMM_WORLD,&req_emfcor_erecv_[l][nb.eid][nb.fi1]);
              }
            }
          }
        }
      }
    }
  }
#endif
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::EnrollFluidBoundaryFunction(enum direction dir,
//                                                       BValFluid_t my_bc)
//  \brief Enroll a user-defined boundary function for fluid

void BoundaryValues::EnrollFluidBoundaryFunction(enum direction dir, BValFluid_t my_bc)
{
  std::stringstream msg;
  if(dir<0 || dir>5) {
    msg << "### FATAL ERROR in EnrollFluidBoundaryCondition function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if(pmy_mblock_->block_bcs[dir]==-1) return;
  if(pmy_mblock_->block_bcs[dir]!=3) {
    msg << "### FATAL ERROR in EnrollFluidBoundaryCondition function" << std::endl
        << "A user-defined boundary condition flag (3) must be specified "
        << "in the input file to use a user-defined boundary function." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  FluidBoundary_[dir]=my_bc;
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::EnrollFieldBoundaryFunction(enum direction dir,
//                                                       BValField_t my_bc)
//  \brief Enroll a user-defined boundary function for magnetic fields

void BoundaryValues::EnrollFieldBoundaryFunction(enum direction dir,BValField_t my_bc)
{
  std::stringstream msg;
  if(dir<0 || dir>5) {
    msg << "### FATAL ERROR in EnrollFieldBoundaryCondition function" << std::endl
        << "dirName = " << dir << " is not valid" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if(pmy_mblock_->block_bcs[dir]==-1) return;
  if(pmy_mblock_->block_bcs[dir]!=3) {
    msg << "### FATAL ERROR in EnrollFieldBoundaryCondition function" << std::endl
        << "A user-defined boundary condition flag (3) must be specified "
        << "in the input file to use a user-defined boundary function." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  FieldBoundary_[dir]=my_bc;
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::CheckBoundary(void)
//  \brief checks if the boundary conditions are correctly enrolled
void BoundaryValues::CheckBoundary(void)
{
  int i;
  MeshBlock *pmb=pmy_mblock_;
  for(int i=0;i<nface_;i++) {
    if(pmb->block_bcs[i]==3) {
      if(FluidBoundary_[i]==NULL) {
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues::CheckBoundary" << std::endl
            << "A user-defined boundary is specified but the fluid boundary function "
            << "is not enrolled in direction " << i  << "." << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
      if (MAGNETIC_FIELDS_ENABLED) {
        if(FieldBoundary_[i]==NULL) {
          std::stringstream msg;
          msg << "### FATAL ERROR in BoundaryValues::CheckBoundary" << std::endl
              << "A user-defined boundary is specified but the field boundary function "
              << "is not enrolled in direction " << i  << "." << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
      }
    }
  }
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::StartReceivingForInit(void)
//  \brief initiate MPI_Irecv for initialization
void BoundaryValues::StartReceivingForInit(void)
{
#ifdef MPI_PARALLEL
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();
  for(int n=0;n<pmb->nneighbor;n++) {
    NeighborBlock& nb=pmb->neighbor[n];
    if(nb.rank!=myrank) { 
      MPI_Start(&req_fluid_recv_[0][nb.bufid]);
      if (MAGNETIC_FIELDS_ENABLED)
        MPI_Start(&req_field_recv_[0][nb.bufid]);
    }
  }
#endif
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::StartReceivingAll(void)
//  \brief initiate MPI_Irecv for all the sweeps
void BoundaryValues::StartReceivingAll(void)
{
#ifdef MPI_PARALLEL
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();
  for(int l=0;l<NSTEP;l++) {
    for(int n=0;n<pmb->nneighbor;n++) {
      NeighborBlock& nb=pmb->neighbor[n];
      if(nb.rank!=myrank) { 
        MPI_Start(&req_fluid_recv_[l][nb.bufid]);
        if(nb.type==neighbor_face && nb.level>mylevel)
          MPI_Start(&req_flcor_recv_[l][nb.fid][nb.fi2][nb.fi1]);
        if (MAGNETIC_FIELDS_ENABLED) {
          MPI_Start(&req_field_recv_[l][nb.bufid]);
          if(nb.type==neighbor_face && nb.level>mylevel)
            MPI_Start(&req_emfcor_frecv_[l][nb.fid][nb.fi2][nb.fi1]);
          if(nb.type==neighbor_edge && nb.level>mylevel && edge_flag_[nb.eid]==true)
            MPI_Start(&req_emfcor_erecv_[l][nb.eid][nb.fi1]);
        }
      }
    }
  }
#endif
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::RestrictFluid(AthenaArray<Real> &src,
//                           int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the fluid data and set them into the coarse buffer
void BoundaryValues::RestrictFluid(AthenaArray<Real> &src, 
                             int csi, int cei, int csj, int cej, int csk, int cek)
{
  MeshBlock *pmb=pmy_mblock_;
  int si=(csi-pmb->cis)*2+pmb->is, ei=(cei-pmb->cis)*2+pmb->is+1;

  // store the restricted data in the prolongation buffer for later use
  if(pmb->block_size.nx3>1) { // 3D
    for (int n=0; n<(NFLUID); ++n) {
      for (int ck=csk; ck<=cek; ck++) {
        int k=(ck-pmb->cks)*2+pmb->ks;
        for (int cj=csj; cj<=cej; cj++) {
          int j=(cj-pmb->cjs)*2+pmb->js;
          pmb->pcoord->CellVolume(k,j,si,ei,fvol_[0][0]);
          pmb->pcoord->CellVolume(k,j+1,si,ei,fvol_[0][1]);
          pmb->pcoord->CellVolume(k+1,j,si,ei,fvol_[1][0]);
          pmb->pcoord->CellVolume(k+1,j+1,si,ei,fvol_[1][1]);
          for (int ci=csi; ci<=cei; ci++) {
            int i=(ci-pmb->cis)*2+pmb->is;
            Real tvol=fvol_[0][0](i)+fvol_[0][0](i+1)+fvol_[0][1](i)+fvol_[0][1](i+1)
                     +fvol_[1][0](i)+fvol_[1][0](i+1)+fvol_[1][1](i)+fvol_[1][1](i+1);
            coarse_cons_(n,ck,cj,ci)=
              (src(n,k  ,j  ,i)*fvol_[0][0](i)+src(n,k  ,j  ,i+1)*fvol_[0][0](i+1)
              +src(n,k  ,j+1,i)*fvol_[0][1](i)+src(n,k  ,j+1,i+1)*fvol_[0][1](i+1)
              +src(n,k+1,j  ,i)*fvol_[1][0](i)+src(n,k+1,j  ,i+1)*fvol_[1][0](i+1)
              +src(n,k+1,j+1,i)*fvol_[1][1](i)+src(n,k+1,j+1,i+1)*fvol_[1][1](i+1))/tvol;
          }
        }
      }
    }
  }
  else if(pmb->block_size.nx2>1) { // 2D
    for (int n=0; n<(NFLUID); ++n) {
      for (int cj=csj; cj<=cej; cj++) {
        int j=(cj-pmb->cjs)*2+pmb->js;
        pmb->pcoord->CellVolume(0,j,si,ei,fvol_[0][0]);
        pmb->pcoord->CellVolume(0,j+1,si,ei,fvol_[0][1]);
        for (int ci=csi; ci<=cei; ci++) {
          int i=(ci-pmb->cis)*2+pmb->is;
          Real tvol=fvol_[0][0](i)+fvol_[0][0](i+1)+fvol_[0][1](i)+fvol_[0][1](i+1);
          coarse_cons_(n,0,cj,ci)=
            (src(n,0,j  ,i)*fvol_[0][0](i)+src(n,0,j  ,i+1)*fvol_[0][0](i+1)
            +src(n,0,j+1,i)*fvol_[0][1](i)+src(n,0,j+1,i+1)*fvol_[0][1](i+1))/tvol;
        }
      }
    }
  }
  else { // 1D
    for (int n=0; n<(NFLUID); ++n) {
      pmb->pcoord->CellVolume(0,0,si,ei,fvol_[0][0]);
      for (int ci=csi; ci<=cei; ci++) {
        int i=(ci-pmb->cis)*2+pmb->is;
        Real tvol=fvol_[0][0](i)+fvol_[0][0](i+1);
        coarse_cons_(n,0,0,ci)
          =(src(n,0,0,i)*fvol_[0][0](i)+src(n,0,0,i+1)*fvol_[0][0](i+1))/tvol;
      }
    }
  }
}


//--------------------------------------------------------------------------------------
//! \fn int BoundaryValues::LoadFluidBoundaryBufferSameLevel(AthenaArray<Real> &src,
//                                                 Real *buf, NeighborBlock& nb)
//  \brief Set fluid boundary buffers for sending to a block on the same level
int BoundaryValues::LoadFluidBoundaryBufferSameLevel(AthenaArray<Real> &src, Real *buf,
                                                     NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;
  int si, sj, sk, ei, ej, ek;

  si=(nb.ox1>0)?(pmb->ie-NGHOST+1):pmb->is;
  ei=(nb.ox1<0)?(pmb->is+NGHOST-1):pmb->ie;
  sj=(nb.ox2>0)?(pmb->je-NGHOST+1):pmb->js;
  ej=(nb.ox2<0)?(pmb->js+NGHOST-1):pmb->je;
  sk=(nb.ox3>0)?(pmb->ke-NGHOST+1):pmb->ks;
  ek=(nb.ox3<0)?(pmb->ks+NGHOST-1):pmb->ke;

  int p=0;
  for (int n=0; n<(NFLUID); ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma simd
        for (int i=si; i<=ei; ++i)
          buf[p++]=src(n,k,j,i);
      }
    }
  }
  return p;
}


//--------------------------------------------------------------------------------------
//! \fn int BoundaryValues::LoadFluidBoundaryBufferToCoarser(AthenaArray<Real> &src,
//                                                 Real *buf, NeighborBlock& nb)
//  \brief Set fluid boundary buffers for sending to a block on the coarser level
int BoundaryValues::LoadFluidBoundaryBufferToCoarser(AthenaArray<Real> &src, Real *buf,
                                                     NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;
  int si, sj, sk, ei, ej, ek;
  int cn=pmb->cnghost-1;

  si=(nb.ox1>0)?(pmb->cie-cn):pmb->cis;
  ei=(nb.ox1<0)?(pmb->cis+cn):pmb->cie;
  sj=(nb.ox2>0)?(pmb->cje-cn):pmb->cjs;
  ej=(nb.ox2<0)?(pmb->cjs+cn):pmb->cje;
  sk=(nb.ox3>0)?(pmb->cke-cn):pmb->cks;
  ek=(nb.ox3<0)?(pmb->cks+cn):pmb->cke;

  // restrict the data before sending
  RestrictFluid(src, si, ei, sj, ej, sk, ek);

  int p=0;

  for (int n=0; n<(NFLUID); ++n) {
    for (int k=sk; k<=ek; k++) {
      for (int j=sj; j<=ej; j++) {
#pragma simd
        for (int i=si; i<=ei; i++)
            buf[p++]=coarse_cons_(n,k,j,i);
      }
    }
  }
  return p;
}


//--------------------------------------------------------------------------------------
//! \fn int BoundaryValues::LoadFluidBoundaryBufferToFiner(AthenaArray<Real> &src,
//                                                 Real *buf, NeighborBlock& nb)
//  \brief Set fluid boundary buffers for sending to a block on the finer level
int BoundaryValues::LoadFluidBoundaryBufferToFiner(AthenaArray<Real> &src, Real *buf,
                                                   NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;
  int si, sj, sk, ei, ej, ek;
  int cn=pmb->cnghost-1;

  si=(nb.ox1>0)?(pmb->ie-cn):pmb->is;
  ei=(nb.ox1<0)?(pmb->is+cn):pmb->ie;
  sj=(nb.ox2>0)?(pmb->je-cn):pmb->js;
  ej=(nb.ox2<0)?(pmb->js+cn):pmb->je;
  sk=(nb.ox3>0)?(pmb->ke-cn):pmb->ks;
  ek=(nb.ox3<0)?(pmb->ks+cn):pmb->ke;

  // send the data first and later prolongate on the target block
  // need to add edges for faces, add corners for edges
  if(nb.ox1==0) {
    if(nb.fi1==1)   si+=pmb->block_size.nx1/2-pmb->cnghost;
    else            ei-=pmb->block_size.nx1/2-pmb->cnghost;
  }
  if(nb.ox2==0 && pmb->block_size.nx2 > 1) {
    if(nb.ox1!=0) {
      if(nb.fi1==1) sj+=pmb->block_size.nx2/2-pmb->cnghost;
      else          ej-=pmb->block_size.nx2/2-pmb->cnghost;
    }
    else {
      if(nb.fi2==1) sj+=pmb->block_size.nx2/2-pmb->cnghost;
      else          ej-=pmb->block_size.nx2/2-pmb->cnghost;
    }
  }
  if(nb.ox3==0 && pmb->block_size.nx3 > 1) {
    if(nb.ox1!=0 && nb.ox2!=0) {
      if(nb.fi1==1) sk+=pmb->block_size.nx3/2-pmb->cnghost;
      else          ek-=pmb->block_size.nx3/2-pmb->cnghost;
    }
    else {
      if(nb.fi2==1) sk+=pmb->block_size.nx3/2-pmb->cnghost;
      else          ek-=pmb->block_size.nx3/2-pmb->cnghost;
    }
  }

  int p=0;
  for (int n=0; n<(NFLUID); ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma simd
        for (int i=si; i<=ei; ++i)
          buf[p++]=src(n,k,j,i);
      }
    }
  }
  return p;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SendFluidBoundaryBuffers(AthenaArray<Real> &src, int step)
//  \brief Send boundary buffers
void BoundaryValues::SendFluidBoundaryBuffers(AthenaArray<Real> &src, int step)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb=pmb->neighbor[n];
    int ssize;
    if(nb.level==mylevel)
      ssize=LoadFluidBoundaryBufferSameLevel(src, fluid_send_[step][nb.bufid],nb);
    else if(nb.level<mylevel)
      ssize=LoadFluidBoundaryBufferToCoarser(src, fluid_send_[step][nb.bufid],nb);
    else
      ssize=LoadFluidBoundaryBufferToFiner(src, fluid_send_[step][nb.bufid], nb);
    if(nb.rank == myrank) { // on the same process
      MeshBlock *pbl=pmb->pmy_mesh->FindMeshBlock(nb.gid);
      std::memcpy(pbl->pbval->fluid_recv_[step][nb.targetid],
                  fluid_send_[step][nb.bufid], ssize*sizeof(Real));
      pbl->pbval->fluid_flag_[step][nb.targetid]=boundary_arrived;
    }
#ifdef MPI_PARALLEL
    else // MPI
      MPI_Start(&req_fluid_send_[step][nb.bufid]);
#endif
  }

  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SetFluidBoundarySameLevel(AthenaArray<Real> &dst,
//                                                     Real *buf, NeighborBlock& nb)
//  \brief Set fluid boundary received from a block on the same level
void BoundaryValues::SetFluidBoundarySameLevel(AthenaArray<Real> &dst, Real *buf,
                                               NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;
  int si, sj, sk, ei, ej, ek;

  if(nb.ox1==0)     si=pmb->is,        ei=pmb->ie;
  else if(nb.ox1>0) si=pmb->ie+1,      ei=pmb->ie+NGHOST;
  else              si=pmb->is-NGHOST, ei=pmb->is-1;
  if(nb.ox2==0)     sj=pmb->js,        ej=pmb->je;
  else if(nb.ox2>0) sj=pmb->je+1,      ej=pmb->je+NGHOST;
  else              sj=pmb->js-NGHOST, ej=pmb->js-1;
  if(nb.ox3==0)     sk=pmb->ks,        ek=pmb->ke;
  else if(nb.ox3>0) sk=pmb->ke+1,      ek=pmb->ke+NGHOST;
  else              sk=pmb->ks-NGHOST, ek=pmb->ks-1;

  int p=0;
  for (int n=0; n<(NFLUID); ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma simd
        for (int i=si; i<=ei; ++i)
          dst(n,k,j,i) = buf[p++];
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SetFluidBoundaryFromCoarser(Real *buf, NeighborBlock& nb)
//  \brief Set fluid prolongation buffer received from a block on the same level
void BoundaryValues::SetFluidBoundaryFromCoarser(Real *buf, NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;

  int si, sj, sk, ei, ej, ek, ll;
  long int lx1, lx2, lx3;
  pmb->uid.GetLocation(lx1,lx2,lx3,ll);
  int cng=pmb->cnghost;

  if(nb.ox1==0) {
    si=pmb->cis, ei=pmb->cie;
    if((lx1&1L)==0L) ei+=cng;
    else             si-=cng; 
  }
  else if(nb.ox1>0)  si=pmb->cie+1,   ei=pmb->cie+cng;
  else               si=pmb->cis-cng, ei=pmb->cis-1;
  if(nb.ox2==0) {
    sj=pmb->cjs, ej=pmb->cje;
    if(pmb->block_size.nx2 > 1) {
      if((lx2&1L)==0L) ej+=cng;
      else             sj-=cng; 
    }
  }
  else if(nb.ox2>0)  sj=pmb->cje+1,   ej=pmb->cje+cng;
  else               sj=pmb->cjs-cng, ej=pmb->cjs-1;
  if(nb.ox3==0) {
    sk=pmb->cks, ek=pmb->cke;
    if(pmb->block_size.nx3 > 1) {
      if((lx3&1L)==0L) ek+=cng;
      else             sk-=cng; 
    }
  }
  else if(nb.ox3>0)  sk=pmb->cke+1,   ek=pmb->cke+cng;
  else               sk=pmb->cks-cng, ek=pmb->cks-1;

  int p=0;
  for (int n=0; n<(NFLUID); ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma simd
        for (int i=si; i<=ei; ++i)
          coarse_cons_(n,k,j,i) = buf[p++];
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SetFluidBoundaryFromFiner(AthenaArray<Real> &dst,
//                                                     Real *buf, NeighborBlock& nb)
//  \brief Set fluid boundary received from a block on the same level
void BoundaryValues::SetFluidBoundaryFromFiner(AthenaArray<Real> &dst, Real *buf,
                                               NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;
  // receive already restricted data
  int si, sj, sk, ei, ej, ek;

  if(nb.ox1==0) {
    si=pmb->is, ei=pmb->ie;
    if(nb.fi1==1)   si+=pmb->block_size.nx1/2;
    else            ei-=pmb->block_size.nx1/2;
  }
  else if(nb.ox1>0) si=pmb->ie+1,      ei=pmb->ie+NGHOST;
  else              si=pmb->is-NGHOST, ei=pmb->is-1;
  if(nb.ox2==0) {
    sj=pmb->js, ej=pmb->je;
    if(pmb->block_size.nx2 > 1) {
      if(nb.ox1!=0) {
        if(nb.fi1==1) sj+=pmb->block_size.nx2/2;
        else          ej-=pmb->block_size.nx2/2;
      }
      else {
        if(nb.fi2==1) sj+=pmb->block_size.nx2/2;
        else          ej-=pmb->block_size.nx2/2;
      }
    }
  }
  else if(nb.ox2>0) sj=pmb->je+1,      ej=pmb->je+NGHOST;
  else              sj=pmb->js-NGHOST, ej=pmb->js-1;
  if(nb.ox3==0) {
    sk=pmb->ks, ek=pmb->ke;
    if(pmb->block_size.nx3 > 1) {
      if(nb.ox1!=0 && nb.ox2!=0) {
        if(nb.fi1==1) sk+=pmb->block_size.nx3/2;
        else          ek-=pmb->block_size.nx3/2;
      }
      else {
        if(nb.fi2==1) sk+=pmb->block_size.nx3/2;
        else          ek-=pmb->block_size.nx3/2;
      }
    }
  }
  else if(nb.ox3>0) sk=pmb->ke+1,      ek=pmb->ke+NGHOST;
  else              sk=pmb->ks-NGHOST, ek=pmb->ks-1;

  int p=0;
  for (int n=0; n<(NFLUID); ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma simd
        for (int i=si; i<=ei; ++i)
          dst(n,k,j,i) = buf[p++];
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn bool BoundaryValues::ReceiveFluidBoundaryBuffers(AthenaArray<Real> &dst, int step)
//  \brief receive the boundary data
bool BoundaryValues::ReceiveFluidBoundaryBuffers(AthenaArray<Real> &dst, int step)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();
  int nc=0;

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb= pmb->neighbor[n];
    if(fluid_flag_[step][nb.bufid]==boundary_completed) { nc++; continue;}
    if(fluid_flag_[step][nb.bufid]==boundary_waiting) {
      if(nb.rank==myrank) // on the same process
        continue;
#ifdef MPI_PARALLEL
      else { // MPI boundary
        int test;
        MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&test,MPI_STATUS_IGNORE);
        MPI_Test(&req_fluid_recv_[step][nb.bufid],&test,MPI_STATUS_IGNORE);
        if(test==false) continue;
        fluid_flag_[step][nb.bufid] = boundary_arrived;
      }
#endif
    }
    if(nb.level==mylevel)
      SetFluidBoundarySameLevel(dst, fluid_recv_[step][nb.bufid], nb);
    else if(nb.level<mylevel) // this set only the prolongation buffer
      SetFluidBoundaryFromCoarser(fluid_recv_[step][nb.bufid], nb);
    else
      SetFluidBoundaryFromFiner(dst, fluid_recv_[step][nb.bufid], nb);

    fluid_flag_[step][nb.bufid] = boundary_completed; // completed
    nc++;
  }

  if(nc<pmb->nneighbor)
    return false;
  return true;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ReceiveFluidBoundaryBuffersWithWait(AthenaArray<Real> &dst,
//                                                               int step)
//  \brief receive the boundary data for initialization
void BoundaryValues::ReceiveFluidBoundaryBuffersWithWait(AthenaArray<Real> &dst, int step)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb= pmb->neighbor[n];
#ifdef MPI_PARALLEL
    if(nb.rank!=myrank)
      MPI_Wait(&req_fluid_recv_[0][nb.bufid],MPI_STATUS_IGNORE);
#endif
    if(nb.level==mylevel)
      SetFluidBoundarySameLevel(dst, fluid_recv_[0][nb.bufid], nb);
    else if(nb.level<mylevel)
      SetFluidBoundaryFromCoarser(fluid_recv_[0][nb.bufid], nb);
    else
      SetFluidBoundaryFromFiner(dst, fluid_recv_[0][nb.bufid], nb);
    fluid_flag_[0][nb.bufid] = boundary_completed; // completed
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SendFluxCorrection(int step)
//  \brief Restrict, pack and send the surace flux to the coarse neighbor(s)
void BoundaryValues::SendFluxCorrection(int step)
{
  MeshBlock *pmb=pmy_mblock_;
  long int lx1, lx2, lx3;
  int mylevel;
  pmb->uid.GetLocation(lx1,lx2,lx3,mylevel);
  int fx1=lx1&1L, fx2=lx2&1L, fx3=lx3&1L;
  int fi1, fi2;

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb= pmb->neighbor[n];
    if(nb.type!=neighbor_face) break;
    if(nb.level==mylevel-1) {
      int p=0;
      // x1 direction
      if(nb.fid==inner_x1 || nb.fid==outer_x1) {
        int i=pmb->is+(pmb->ie-pmb->is+1)*nb.fid;
        fi1=fx2, fi2=fx3;
        if(pmb->block_size.nx3>1) { // 3D
          for(int nn=0; nn<NFLUID; nn++) {
            for(int k=pmb->ks; k<=pmb->ke; k+=2) {
              for(int j=pmb->js; j<=pmb->je; j+=2) {
                Real amm=pmb->pcoord->GetFace1Area(k,   j,   i);
                Real amp=pmb->pcoord->GetFace1Area(k,   j+1, i);
                Real apm=pmb->pcoord->GetFace1Area(k+1, j,   i);
                Real app=pmb->pcoord->GetFace1Area(k+1, j+1, i);
                Real tarea=amm+amp+apm+app;
                flcor_send_[step][nb.fid][p++]=
                           (surface_flux_[nb.fid](nn, k  , j  )*amm
                           +surface_flux_[nb.fid](nn, k  , j+1)*amp
                           +surface_flux_[nb.fid](nn, k+1, j  )*apm
                           +surface_flux_[nb.fid](nn, k+1, j+1)*app)/tarea;
              }
            }
          }
        }
        else if(pmb->block_size.nx2>1) { // 2D
          for(int nn=0; nn<NFLUID; nn++) {
            for(int j=pmb->js; j<=pmb->je; j+=2) {
              Real am=pmb->pcoord->GetFace1Area(0, j,   i);
              Real ap=pmb->pcoord->GetFace1Area(0, j+1, i);
              Real tarea=am+ap;
              flcor_send_[step][nb.fid][p++]=
                         (surface_flux_[nb.fid](nn, 0, j  )*am
                         +surface_flux_[nb.fid](nn, 0, j+1)*ap)/tarea;
            }
          }
        }
        else { // 1D
          for(int nn=0; nn<NFLUID; nn++)
            flcor_send_[step][nb.fid][p++]=surface_flux_[nb.fid](nn, 0, 0);
        }
      }
      // x2 direction
      else if(nb.fid==inner_x2 || nb.fid==outer_x2) {
        int j=pmb->js+(pmb->je-pmb->js+1)*(nb.fid&1);
        fi1=fx1, fi2=fx3;
        if(pmb->block_size.nx3>1) { // 3D
          for(int nn=0; nn<NFLUID; nn++) {
            for(int k=pmb->ks; k<=pmb->ke; k+=2) {
              pmb->pcoord->Face2Area(k  , j, pmb->is, pmb->ie, sarea_[0]);
              pmb->pcoord->Face2Area(k+1, j, pmb->is, pmb->ie, sarea_[1]);
              for(int i=pmb->is; i<=pmb->ie; i+=2) {
                Real tarea=sarea_[0](i)+sarea_[0](i+1)+sarea_[1](i)+sarea_[1](i+1);
                flcor_send_[step][nb.fid][p++]=
                           (surface_flux_[nb.fid](nn, k  , i  )*sarea_[0](i  )
                           +surface_flux_[nb.fid](nn, k  , i+1)*sarea_[0](i+1)
                           +surface_flux_[nb.fid](nn, k+1, i  )*sarea_[1](i  )
                           +surface_flux_[nb.fid](nn, k+1, i+1)*sarea_[1](i+1))/tarea;
              }
            }
          }
        }
        else if(pmb->block_size.nx2>1) { // 2D
          for(int nn=0; nn<NFLUID; nn++) {
            pmb->pcoord->Face2Area(0, j, pmb->is ,pmb->ie, sarea_[0]);
            for(int i=pmb->is; i<=pmb->ie; i+=2) {
              Real tarea=sarea_[0](i)+sarea_[0](i+1);
              flcor_send_[step][nb.fid][p++]=
                         (surface_flux_[nb.fid](nn, 0, i  )*sarea_[0](i  )
                         +surface_flux_[nb.fid](nn, 0, i+1)*sarea_[0](i+1))/tarea;
            }
          }
        }
      }
      // x3 direction - 3D only
      else if(nb.fid==inner_x3 || nb.fid==outer_x3) {
        int k=pmb->ks+(pmb->ke-pmb->ks+1)*(nb.fid&1);
        fi1=fx1, fi2=fx2;
        for(int nn=0; nn<NFLUID; nn++) {
          for(int j=pmb->js; j<=pmb->je; j+=2) {
            pmb->pcoord->Face3Area(k, j,   pmb->is, pmb->ie, sarea_[0]);
            pmb->pcoord->Face3Area(k, j+1, pmb->is, pmb->ie, sarea_[1]);
            for(int i=pmb->is; i<=pmb->ie; i+=2) {
              Real tarea=sarea_[0](i)+sarea_[0](i+1)+sarea_[1](i)+sarea_[1](i+1);
              flcor_send_[step][nb.fid][p++]=
                         (surface_flux_[nb.fid](nn, j  , i  )*sarea_[0](i  )
                         +surface_flux_[nb.fid](nn, j  , i+1)*sarea_[0](i+1)
                         +surface_flux_[nb.fid](nn, j+1, i  )*sarea_[1](i  )
                         +surface_flux_[nb.fid](nn, j+1, i+1)*sarea_[1](i+1))/tarea;
            }
          }
        }
      }
      if(nb.rank==myrank) { // on the same node
        MeshBlock *pbl=pmb->pmy_mesh->FindMeshBlock(nb.gid);
        std::memcpy(pbl->pbval->flcor_recv_[step][(nb.fid^1)][fi2][fi1],
                    flcor_send_[step][nb.fid], p*sizeof(Real));
        pbl->pbval->flcor_flag_[step][(nb.fid^1)][fi2][fi1]=boundary_arrived;
      }
#ifdef MPI_PARALLEL
      else
        MPI_Start(&req_flcor_send_[step][nb.fid]);
#endif
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn bool BoundaryValues::ReceiveFluxCorrection(AthenaArray<Real> &dst, int step)
//  \brief Receive and apply the surace flux from the finer neighbor(s)
bool BoundaryValues::ReceiveFluxCorrection(AthenaArray<Real> &dst, int step)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();
  int nc=0, nff=0;
  Real dt=pmb->pmy_mesh->dt;
  if(step==1) dt*=0.5;

  // count the number of finer faces.
  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb= pmb->neighbor[n];
    if(nb.type==neighbor_face && nb.level==mylevel+1) nff++;
    if(nb.type!=neighbor_face) break;
  }

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb= pmb->neighbor[n];
    if(nb.type!=neighbor_face) break;
    if(nb.level==mylevel+1) {
      if(flcor_flag_[step][nb.fid][nb.fi2][nb.fi1]==boundary_completed) { nc++; continue; }
      if(flcor_flag_[step][nb.fid][nb.fi2][nb.fi1]==boundary_waiting) {
        if(nb.rank==myrank) // on the same process
          continue;
#ifdef MPI_PARALLEL
        else { // MPI boundary
          int test;
          MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&test,MPI_STATUS_IGNORE);
          MPI_Test(&req_flcor_recv_[step][nb.fid][nb.fi2][nb.fi1],&test,MPI_STATUS_IGNORE);
          if(test==false) continue;
          flcor_flag_[step][nb.fid][nb.fi2][nb.fi1] = boundary_arrived;
        }
#endif
      }
      // boundary arrived; apply flux correction
      Real *buf=flcor_recv_[step][nb.fid][nb.fi2][nb.fi1];
      int p=0;
      if(nb.fid==inner_x1 || nb.fid==outer_x1) {
        int ic=pmb->is+(pmb->ie-pmb->is)*nb.fid;
        int is=ic+nb.fid;
        int js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
        if(nb.fi1==0) je-=pmb->block_size.nx2/2;
        else          js+=pmb->block_size.nx2/2;
        if(nb.fi2==0) ke-=pmb->block_size.nx3/2;
        else          ks+=pmb->block_size.nx3/2;
        Real sign=(Real)(nb.fid*2-1); // -1 for inner, +1 for outer
        for(int nn=0; nn<NFLUID; nn++) {
          for(int k=ks; k<=ke; k++) {
            for(int j=js; j<=je; j++) {
              Real area=pmb->pcoord->GetFace1Area(k,j,is);
              Real vol=pmb->pcoord->GetCellVolume(k,j,ic);
              dst(nn,k,j,ic)+=sign*dt*area*(surface_flux_[nb.fid](nn,k,j)-buf[p++])/vol;
            }
          }
        }
      }
      else if(nb.fid==inner_x2 || nb.fid==outer_x2) {
        int jc=pmb->js+(pmb->je-pmb->js)*(nb.fid&1);
        int js=jc+(nb.fid&1);
        int is=pmb->is, ie=pmb->ie, ks=pmb->ks, ke=pmb->ke;
        if(nb.fi1==0) ie-=pmb->block_size.nx1/2;
        else          is+=pmb->block_size.nx1/2;
        if(nb.fi2==0) ke-=pmb->block_size.nx3/2;
        else          ks+=pmb->block_size.nx3/2;
        Real sign=(Real)((nb.fid&1)*2-1); // -1 for inner, +1 for outer
        for(int nn=0; nn<NFLUID; nn++) {
          for(int k=ks; k<=ke; k++) {
            pmb->pcoord->Face2Area(k,js,is,ie,sarea_[0]);
            pmb->pcoord->CellVolume(k,jc,is,ie,fvol_[0][0]);
            for(int i=is; i<=ie; i++)
              dst(nn,k,jc,i)+=dt*sign*sarea_[0](i)
                            *(surface_flux_[nb.fid](nn,k,i)-buf[p++])/fvol_[0][0](i);
          }
        }
      }
      else if(nb.fid==inner_x3 || nb.fid==outer_x3) {
        int kc=pmb->ks+(pmb->ke-pmb->ks)*(nb.fid&1);
        int ks=kc+(nb.fid&1);
        int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je;
        if(nb.fi1==0) ie-=pmb->block_size.nx1/2;
        else          is+=pmb->block_size.nx1/2;
        if(nb.fi2==0) je-=pmb->block_size.nx2/2;
        else          js+=pmb->block_size.nx2/2;
        Real sign=(Real)((nb.fid&1)*2-1); // -1 for inner, +1 for outer
        for(int nn=0; nn<NFLUID; nn++) {
          for(int j=js; j<=je; j++) {
            pmb->pcoord->Face3Area(ks,j,is,ie,sarea_[0]);
            pmb->pcoord->CellVolume(kc,j,is,ie,fvol_[0][0]);
            for(int i=is; i<=ie; i++)
              dst(nn,kc,j,i)+=dt*sign*sarea_[0](i)
                            *(surface_flux_[nb.fid](nn,j,i)-buf[p++])/fvol_[0][0](i);
          }
        }
      }

      flcor_flag_[step][nb.fid][nb.fi2][nb.fi1] = boundary_completed;
      nc++;
    }
  }

  if(nc<nff)
    return false;
  return true;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ProlongateFluidBoundaries(AthenaArray<Real> &dst)
//  \brief Prolongate the ghost zones from the prolongation buffer
void BoundaryValues::ProlongateFluidBoundaries(AthenaArray<Real> &dst)
{
  MeshBlock *pmb=pmy_mblock_;
  Coordinates *pco=pmb->pcoord;
  int mylevel;
  long int lx1, lx2, lx3;
  pmb->uid.GetLocation(lx1, lx2, lx3, mylevel);
  if(pmb->block_size.nx2>1) {  // only in 2D or 3D
    for(int n=0; n<pmb->nneighbor; n++) {
      NeighborBlock& nb= pmb->neighbor[n];
      if(nb.level >= mylevel) continue;
      // fill the required ghost-ghost zone
      int nis, nie, njs, nje, nks, nke;
      nis=std::max(nb.ox1-1,-1), nie=std::min(nb.ox1+1,1);
      njs=std::max(nb.ox2-1,-1), nje=std::min(nb.ox2+1,1);
      if(pmb->block_size.nx3==1) nks=0, nke=0;
      else nks=std::max(nb.ox3-1,-1), nke=std::min(nb.ox3+1,1);
      for(int nk=nks; nk<=nke; nk++) {
        for(int nj=njs; nj<=nje; nj++) {
          for(int ni=nis; ni<=nie; ni++) {
            if(ni==0 && nj==0 && nk==0) continue; // skip myself
            if(pmb->nblevel[nk+1][nj+1][ni+1]!=mylevel
            && pmb->nblevel[nk+1][nj+1][ni+1]!=-1)
              continue; // physical boundary will also be restricted

            // this neighbor block is on the same level
            // and needs to be restricted for prolongation
            int ris, rie, rjs, rje, rks, rke;
            if(ni==0) {
              ris=pmb->cis, rie=pmb->cie;
              if(nb.ox1==1) ris=pmb->cie;
              else if(nb.ox1==-1) rie=pmb->cis;
            }
            else if(ni== 1) ris=pmb->cie+1, rie=pmb->cie+1;
            else if(ni==-1) ris=pmb->cis-1, rie=pmb->cis-1;
            if(nj==0) {
              rjs=pmb->cjs, rje=pmb->cje;
              if(nb.ox2==1) rjs=pmb->cje;
              else if(nb.ox2==-1) rje=pmb->cjs;
            }
            else if(nj== 1) rjs=pmb->cje+1, rje=pmb->cje+1;
            else if(nj==-1) rjs=pmb->cjs-1, rje=pmb->cjs-1;
            if(nk==0) {
              rks=pmb->cks, rke=pmb->cke;
              if(nb.ox3==1) rks=pmb->cke;
              else if(nb.ox3==-1) rke=pmb->cks;
            }
            else if(nk== 1) rks=pmb->cke+1, rke=pmb->cke+1;
            else if(nk==-1) rks=pmb->cks-1, rke=pmb->cks-1;
            RestrictFluid(dst, ris, rie, rjs, rje, rks, rke);
          }
        }
      }

      // now that the ghost-ghost zones are filled
      // calculate the slope with a limiter and interpolate the data
      int cn = (NGHOST+1)/2;
      int si, ei, sj, ej, sk, ek;
      if(nb.ox1==0) {
        si=pmb->cis, ei=pmb->cie;
        if((lx1&1L)==0L) ei++;
        else             si--;
      }
      else if(nb.ox1>0) si=pmb->cie+1,  ei=pmb->cie+cn;
      else              si=pmb->cis-cn, ei=pmb->cis-1;
      if(nb.ox2==0) {
        sj=pmb->cjs, ej=pmb->cje;
        if((lx2&1L)==0L) ej++;
        else             sj--;
      }
      else if(nb.ox2>0) sj=pmb->cje+1,  ej=pmb->cje+cn;
      else              sj=pmb->cjs-cn, ej=pmb->cjs-1;
      if(nb.ox3==0) {
        sk=pmb->cks, ek=pmb->cke;
        if((lx3&1L)==0L) ek++;
        else             sk--;
      }
      else if(nb.ox3>0) sk=pmb->cke+1,  ek=pmb->cke+cn;
      else              sk=pmb->cks-cn, ek=pmb->cks-1;

      if(pmb->block_size.nx3 > 1) { // 3D
        for(int n=0; n<NFLUID; n++) {
          for(int k=sk; k<=ek; k++) {
            Real& x3m = pco->coarse_x3v(k-1);
            Real& x3c = pco->coarse_x3v(k);
            Real& x3p = pco->coarse_x3v(k+1);
            Real& x3fm = pco->coarse_x3f(k);
            Real& x3fp = pco->coarse_x3f(k+1);
            Real& dx3m = pco->coarse_x3v(k  )-pco->coarse_x3v(k-1);
            Real& dx3p = pco->coarse_x3v(k+1)-pco->coarse_x3v(k  );
            int fk=(k-pmb->cks)*2+pmb->ks;
            Real& fx3m = pco->x3v(fk);
            Real& fx3p = pco->x3v(fk+1);
            Real dx3fm= x3c-fx3m;
            Real dx3fp= fx3p-x3c;
            for(int j=sj; j<=ej; j++) {
              Real& x2m = pco->coarse_x2v(j-1);
              Real& x2c = pco->coarse_x2v(j);
              Real& x2p = pco->coarse_x2v(j+1);
              Real& x2fm = pco->coarse_x2f(j);
              Real& x2fp = pco->coarse_x2f(j+1);
              Real& dx2m = pco->coarse_x2v(j  )-pco->coarse_x2v(j-1);
              Real& dx2p = pco->coarse_x2v(j+1)-pco->coarse_x2v(j  );
              int fj=(j-pmb->cjs)*2+pmb->js;
              Real& fx2m = pco->x2v(fj);
              Real& fx2p = pco->x2v(fj+1);
              Real dx2fm= x2c-fx2m;
              Real dx2fp= fx2p-x2c;
              for(int i=si; i<=ei; i++) {
                Real& x1m = pco->coarse_x1v(i-1);
                Real& x1c = pco->coarse_x1v(i);
                Real& x1p = pco->coarse_x1v(i+1);
                Real& x1fm = pco->coarse_x1f(i);
                Real& x1fp = pco->coarse_x1f(i+1);
                Real& dx1m = pco->coarse_x1v(i  )-pco->coarse_x1v(i-1);
                Real& dx1p = pco->coarse_x1v(i+1)-pco->coarse_x1v(i  );
                int fi=(i-pmb->cis)*2+pmb->is;
                Real& fx1m = pco->x1v(fi);
                Real& fx1p = pco->x1v(fi+1);
                Real dx1fm= x1c-fx1m;
                Real dx1fp= fx1p-x1c;
                Real ccval=coarse_cons_(n,k,j,i);
                // calculate 3D gradients using Mignone 2014's modified van-Leer limiter
                Real gx1m = (ccval-coarse_cons_(n,k,j,i-1))/dx1m;
                Real gx1p = (coarse_cons_(n,k,j,i+1)-ccval)/dx1p;
                Real gx1c = gx1m*gx1p;
                if(gx1c>0.0) {
                  Real cf=dx1p/(x1fp-x1c);
                  Real cb=dx1m/(x1c-x1fm);
                  gx1c=gx1c*(cf*gx1m+cb*gx1p)/(gx1m*gx1m+(cf+cb-2.0)*gx1c+gx1p*gx1p);
                }
                else gx1c=0.0;

                Real gx2m = (ccval-coarse_cons_(n,k,j-1,i))/dx2m;
                Real gx2p = (coarse_cons_(n,k,j+1,i)-ccval)/dx2p;
                Real gx2c = gx2m*gx2p;
                if(gx2c>0.0) {
                  Real cf=dx2p/(x2fp-x2c);
                  Real cb=dx2m/(x2c-x2fm);
                  gx2c=gx2c*(cf*gx2m+cb*gx2p)/(gx2m*gx2m+(cf+cb-2.0)*gx2c+gx2p*gx2p);
                }
                else gx2c=0.0;
  
                Real gx3m = (ccval-coarse_cons_(n,k-1,j,i))/dx3m;
                Real gx3p = (coarse_cons_(n,k+1,j,i)-ccval)/dx3p;
                Real gx3c = gx3m*gx3p;
                if(gx3c>0.0) {
                  Real cf=dx3p/(x3fp-x3c);
                  Real cb=dx3m/(x3c-x3fm);
                  gx3c=gx3c*(cf*gx3m+cb*gx3p)/(gx3m*gx3m+(cf+cb-2.0)*gx3c+gx3p*gx3p);
                }
                else gx3c=0.0;

                // interpolate onto the finer grid
                dst(n,fk  ,fj  ,fi  )=ccval-gx1c*dx1fm-gx2c*dx2fm-gx3c*dx3fm;
                dst(n,fk  ,fj  ,fi+1)=ccval+gx1c*dx1fp-gx2c*dx2fm-gx3c*dx3fm;
                dst(n,fk  ,fj+1,fi  )=ccval-gx1c*dx1fm+gx2c*dx2fp-gx3c*dx3fm;
                dst(n,fk  ,fj+1,fi+1)=ccval+gx1c*dx1fp+gx2c*dx2fp-gx3c*dx3fm;
                dst(n,fk+1,fj  ,fi  )=ccval-gx1c*dx1fm-gx2c*dx2fm+gx3c*dx3fp;
                dst(n,fk+1,fj  ,fi+1)=ccval+gx1c*dx1fp-gx2c*dx2fm+gx3c*dx3fp;
                dst(n,fk+1,fj+1,fi  )=ccval-gx1c*dx1fm+gx2c*dx2fp+gx3c*dx3fp;
                dst(n,fk+1,fj+1,fi+1)=ccval+gx1c*dx1fp+gx2c*dx2fp+gx3c*dx3fp;
              }
            }
          }
        }
      }
      else if(pmb->block_size.nx2 > 1) { // 2D
        int k=sk, fk=sk;
        for(int n=0; n<NFLUID; n++) {
          for(int j=sj; j<=ej; j++) {
            Real& x2m = pco->coarse_x2v(j-1);
            Real& x2c = pco->coarse_x2v(j);
            Real& x2p = pco->coarse_x2v(j+1);
            Real& x2fm = pco->coarse_x2f(j);
            Real& x2fp = pco->coarse_x2f(j+1);
            Real& dx2m = pco->coarse_x2v(j  )-pco->coarse_x2v(j-1);
            Real& dx2p = pco->coarse_x2v(j+1)-pco->coarse_x2v(j  );
            int fj=(sj-pmb->cjs)*2+NGHOST;
            Real& fx2m = pco->x2v(fj);
            Real& fx2p = pco->x2v(fj+1);
            Real dx2fm= x2c-fx2m;
            Real dx2fp= fx2p-x2c;
            for(int i=si; i<=ei; i++) {
              Real& x1m = pco->coarse_x1v(i-1);
              Real& x1c = pco->coarse_x1v(i);
              Real& x1p = pco->coarse_x1v(i+1);
              Real& x1fm = pco->coarse_x1f(i);
              Real& x1fp = pco->coarse_x1f(i+1);
              Real& dx1m = pco->coarse_x1v(i  )-pco->coarse_x1v(i-1);
              Real& dx1p = pco->coarse_x1v(i+1)-pco->coarse_x1v(i  );
              int fi=(si-pmb->cis)*2+NGHOST;
              Real& fx1m = pco->x1v(fi);
              Real& fx1p = pco->x1v(fi+1);
              Real dx1fm= x1c-fx1m;
              Real dx1fp= fx1p-x1c;
              Real ccval=coarse_cons_(n,k,j,i);
              // calculate 2D gradients using Mignone 2014's modified van-Leer limiter
              Real gx1m = (ccval-coarse_cons_(n,k,j,i-1))/dx1m;
              Real gx1p = (coarse_cons_(n,k,j,i+1)-ccval)/dx1p;
              Real gx1c = gx1m*gx1p;
              if(gx1c>0.0) {
                Real cf=dx1p/(x1fp-x1c);
                Real cb=dx1m/(x1c-x1fm);
                gx1c=gx1c*(cf*gx1m+cb*gx1p)/(gx1m*gx1m+(cf+cb-2.0)*gx1c+gx1p*gx1p);
              }
              else gx1c=0.0;

              Real gx2m = (ccval-coarse_cons_(n,k,j-1,i))/dx2m;
              Real gx2p = (coarse_cons_(n,k,j+1,i)-ccval)/dx2p;
              Real gx2c = gx2m*gx2p;
              if(gx2c>0.0) {
                Real cf=dx2p/(x2fp-x2c);
                Real cb=dx2m/(x2c-x2fm);
                gx2c=gx2c*(cf*gx2m+cb*gx2p)/(gx2m*gx2m+(cf+cb-2.0)*gx2c+gx2p*gx2p);
              }
              else gx2c=0.0;

              // interpolate on to the finer grid
              dst(n,fk  ,fj  ,fi  )=ccval-gx1c*dx1fm-gx2c*dx2fm;
              dst(n,fk  ,fj  ,fi+1)=ccval+gx1c*dx1fp-gx2c*dx2fm;
              dst(n,fk  ,fj+1,fi  )=ccval-gx1c*dx1fm+gx2c*dx2fp;
              dst(n,fk  ,fj+1,fi+1)=ccval+gx1c*dx1fp+gx2c*dx2fp;
            }
          }
        }
      }
      else { // 1D
        int k=sk, fk=sk, j=sj, fj=sj;
        for(int n=0; n<NFLUID; n++) {
          for(int i=si; i<=ei; i++) {
            Real& x1m = pco->coarse_x1v(i-1);
            Real& x1c = pco->coarse_x1v(i);
            Real& x1p = pco->coarse_x1v(i+1);
            Real& x1fm = pco->coarse_x1f(i);
            Real& x1fp = pco->coarse_x1f(i+1);
            Real& dx1m = pco->coarse_x1v(i  )-pco->coarse_x1v(i-1);
            Real& dx1p = pco->coarse_x1v(i+1)-pco->coarse_x1v(i  );
            int fi=(si-pmb->cis)*2+NGHOST;
            Real& fx1m = pco->x1v(fi);
            Real& fx1p = pco->x1v(fi+1);
            Real dx1fm= x1c-fx1m;
            Real dx1fp= fx1p-x1c;
            Real ccval=coarse_cons_(n,k,j,i);
            // calculate 1D gradient using Mignone 2014's modified van-Leer limiter
            Real gx1m = (ccval-coarse_cons_(n,k,j,i-1))/dx1m;
            Real gx1p = (coarse_cons_(n,k,j,i+1)-ccval)/dx1p;
            Real gx1c = gx1m*gx1p;
            if(gx1c>0.0) {
              Real cf=dx1p/(x1fp-x1c);
              Real cb=dx1m/(x1c-x1fm);
              gx1c=gx1c*(cf*gx1m+cb*gx1p)/(gx1m*gx1m+(cf+cb-2.0)*gx1c+gx1p*gx1p);
            }
            else gx1c=0.0;

            // interpolate on to the finer grid
            dst(n,fk  ,fj  ,fi  )=ccval-gx1c*dx1fm;
            dst(n,fk  ,fj  ,fi+1)=ccval+gx1c*dx1fp;
          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::RestrictFieldX1(AthenaArray<Real> &bx1f,
//                           int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x1 field data and set them into the coarse buffer
void BoundaryValues::RestrictFieldX1(AthenaArray<Real> &bx1f, 
                             int csi, int cei, int csj, int cej, int csk, int cek)
{
  MeshBlock *pmb=pmy_mblock_;
  int si=(csi-pmb->cis)*2+pmb->is, ei=(cei-pmb->cis)*2+pmb->is;

  // store the restricted data in the prolongation buffer for later use
  if(pmb->block_size.nx3>1) { // 3D
    for (int ck=csk; ck<=cek; ck++) {
      int k=(ck-pmb->cks)*2+pmb->ks;
      for (int cj=csj; cj<=cej; cj++) {
        int j=(cj-pmb->cjs)*2+pmb->js;
        // reuse fvol_ arrays as surface area
        pmb->pcoord->Face1Area(k,   j,   si, ei, fvol_[0][0]);
        pmb->pcoord->Face1Area(k,   j+1, si, ei, fvol_[0][1]);
        pmb->pcoord->Face1Area(k+1, j,   si, ei, fvol_[1][0]);
        pmb->pcoord->Face1Area(k+1, j+1, si, ei, fvol_[1][1]);
        for (int ci=csi; ci<=cei; ci++) {
          int i=(ci-pmb->cis)*2+pmb->is;
          Real tarea=fvol_[0][0](i)+fvol_[0][1](i)+fvol_[1][0](i)+fvol_[1][1](i);
          coarse_b_.x1f(ck,cj,ci)=
            (bx1f(k  ,j,i)*fvol_[0][0](i)+bx1f(k  ,j+1,i)*fvol_[0][1](i)
            +bx1f(k+1,j,i)*fvol_[1][0](i)+bx1f(k+1,j+1,i)*fvol_[1][1](i))/tarea;
        }
      }
    }
  }
  else if(pmb->block_size.nx2>1) { // 2D
    int k=pmb->ks
    for (int cj=csj; cj<=cej; cj++) {
      int j=(cj-pmb->cjs)*2+pmb->js;
      // reuse fvol_ arrays as surface area
      pmb->pcoord->Face1Area(k,  j,   si, ei, fvol_[0][0]);
      pmb->pcoord->Face1Area(k,  j+1, si, ei, fvol_[0][1]);
      for (int ci=csi; ci<=cei; ci++) {
        int i=(ci-pmb->cis)*2+pmb->is;
        Real tarea=fvol_[0][0](i)+fvol_[0][1](i);
        coarse_b_.x1f(csk,cj,ci)=
          (bx1f(k,j,i)*fvol_[0][0](i)+bx1f(k,j+1,i)*fvol_[0][1](i))/tarea;
      }
    }
  }
  else { // 1D - no restriction, just copy 
    for (int ci=csi; ci<=cei; ci++) {
      int i=(ci-pmb->cis)*2+pmb->is;
      coarse_b_.x1f(csk,csj,ci)=bx1f(pmb->csk,pmb->csj,i);
    }
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::RestrictFieldX2(AthenaArray<Real> &bx2f,
//                           int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x2 field data and set them into the coarse buffer
void BoundaryValues::RestrictFieldX2(AthenaArray<Real> &bx2f, 
                             int csi, int cei, int csj, int cej, int csk, int cek)
{
  MeshBlock *pmb=pmy_mblock_;
  int si=(csi-pmb->cis)*2+pmb->is, ei=(cei-pmb->cis)*2+pmb->is+1;

  // store the restricted data in the prolongation buffer for later use
  if(pmb->block_size.nx3>1) { // 3D
    for (int ck=csk; ck<=cek; ck++) {
      int k=(ck-pmb->cks)*2+pmb->ks;
      for (int cj=csj; cj<=cej; cj++) {
        int j=(cj-pmb->cjs)*2+pmb->js;
        // reuse fvol_ arrays as surface area
        pmb->pcoord->Face2Area(k,   j,  si, ei, fvol_[0][0]);
        pmb->pcoord->Face2Area(k+1, j,  si, ei, fvol_[0][1]);
        for (int ci=csi; ci<=cei; ci++) {
          int i=(ci-pmb->cis)*2+pmb->is;
          Real tarea=fvol_[0][0](i)+fvol_[0][0](i+1)+fvol_[0][1](i)+fvol_[0][1](i+1);
          coarse_b_.x2f(ck,cj,ci)=
            (bx2f(k  ,j,i)*fvol_[0][0](i)+bx2f(k  ,j,i+1)*fvol_[0][0](i+1)
            +bx2f(k+1,j,i)*fvol_[0][1](i)+bx2f(k+1,j,i+1)*fvol_[0][1](i+1))/tarea;
        }
      }
    }
  }
  else if(pmb->block_size.nx2>1) { // 2D
    int k=pmb->ks
    for (int cj=csj; cj<=cej; cj++) {
      int j=(cj-pmb->cjs)*2+pmb->js;
      // reuse fvol_ arrays as surface area
      pmb->pcoord->Face2Area(k, j, si, ei, fvol_[0][0]);
      for (int ci=csi; ci<=cei; ci++) {
        int i=(ci-pmb->cis)*2+pmb->is;
        Real tarea=fvol_[0][0](i)+fvol_[0][0](i+1);
        coarse_b_.x2f(pmb->cks,cj,ci)=
          (bx2f(k,j,i)*fvol_[0][0](i)+bx2f(k,j,i+1)*fvol_[0][0](i+1))/tarea;
      }
    }
  }
  else { // 1D - no restriction, just copy 
    int k=pmb->ks, j=pmb->js;
    pmb->pcoord->Face2Area(k, j, si, ei, fvol_[0][0]);
    for (int ci=csi; ci<=cei; ci++) {
      int i=(ci-pmb->cis)*2+pmb->is;
        Real tarea=fvol_[0][0](i)+fvol_[0][0](i+1);
        coarse_b_.x2f(pmb->cks,pmb->cjs,ci)=
          (bx2f(k,j,i)*fvol_[0][0](i)+bx2f(k,j,i+1)*fvol_[0][0](i+1))/tarea;
    }
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::RestrictFieldX3(AthenaArray<Real> &bx3f,
//                           int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x3 field data and set them into the coarse buffer
void BoundaryValues::RestrictFieldX3(AthenaArray<Real> &bx3f, 
                             int csi, int cei, int csj, int cej, int csk, int cek)
{
  MeshBlock *pmb=pmy_mblock_;
  int si=(csi-pmb->cis)*2+pmb->is, ei=(cei-pmb->cis)*2+pmb->is+1;

  // store the restricted data in the prolongation buffer for later use
  if(pmb->block_size.nx3>1) { // 3D
    for (int ck=csk; ck<=cek; ck++) {
      int k=(ck-pmb->cks)*2+pmb->ks;
      for (int cj=csj; cj<=cej; cj++) {
        int j=(cj-pmb->cjs)*2+pmb->js;
        // reuse fvol_ arrays as surface area
        pmb->pcoord->Face3Area(k,   j,  si, ei, fvol_[0][0]);
        pmb->pcoord->Face3Area(k, j+1,  si, ei, fvol_[0][1]);
        for (int ci=csi; ci<=cei; ci++) {
          int i=(ci-pmb->cis)*2+pmb->is;
          Real tarea=fvol_[0][0](i)+fvol_[0][0](i+1)+fvol_[0][1](i)+fvol_[0][1](i+1);
          coarse_b_.x3f(ck,cj,ci)=
            (bx3f(k,j  ,i)*fvol_[0][0](i)+bx3f(k,j  ,i+1)*fvol_[0][0](i+1)
            +bx3f(k,j+1,i)*fvol_[0][1](i)+bx3f(k,j+1,i+1)*fvol_[0][1](i+1))/tarea;
        }
      }
    }
  }
  else if(pmb->block_size.nx2>1) { // 2D
    int k=pmb->ks
    for (int cj=csj; cj<=cej; cj++) {
      int j=(cj-pmb->cjs)*2+pmb->js;
      // reuse fvol_ arrays as surface area
      pmb->pcoord->Face3Area(k,   j, si, ei, fvol_[0][0]);
      pmb->pcoord->Face3Area(k, j+1, si, ei, fvol_[0][1]);
      for (int ci=csi; ci<=cei; ci++) {
        int i=(ci-pmb->cis)*2+pmb->is;
        Real tarea=fvol_[0][0](i)+fvol_[0][0](i+1)+fvol_[0][1](i)+fvol_[0][1](i+1);
        coarse_b_.x3f(pmb->cks,cj,ci)=
            (bx3f(k,j  ,i)*fvol_[0][0](i)+bx3f(k,j  ,i+1)*fvol_[0][0](i+1)
            +bx3f(k,j+1,i)*fvol_[0][1](i)+bx3f(k,j+1,i+1)*fvol_[0][1](i+1))/tarea;
      }
    }
  }
  else { // 1D - no restriction, just copy 
    int k=pmb->ks, j=pmb->js;
    pmb->pcoord->Face3Area(k, j, si, ei, fvol_[0][0]);
    for (int ci=csi; ci<=cei; ci++) {
      int i=(ci-pmb->cis)*2+pmb->is;
        Real tarea=fvol_[0][0](i)+fvol_[0][0](i+1);
        coarse_b_.x3f(pmb->cks,pmb->cjs,ci)=
          (bx3f(k,j,i)*fvol_[0][0](i)+bx3f(k,j,i+1)*fvol_[0][0](i+1))/tarea;
    }
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn int BoundaryValues::LoadFieldBoundaryBufferSameLevel(InterfaceField &src,
//                                                 Real *buf, NeighborBlock& nb)
//  \brief Set field boundary buffers for sending to a block on the same level
int BoundaryValues::LoadFieldBoundaryBufferSameLevel(InterfaceField &src, Real *buf,
                                                     NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;
  int si, sj, sk, ei, ej, ek;
  int p=0;

  // bx1
  if(nb.ox1==0)     si=pmb->is,          ei=pmb->ie+1;
  else if(nb.ox1>0) si=pmb->ie-NGHOST+1, ei=pmb->ie;
  else              si=pmb->is+1,        ei=pmb->is+NGHOST;
  if(nb.ox2==0)     sj=pmb->js,          ej=pmb->je;
  else if(nb.ox2>0) sj=pmb->je-NGHOST+1, ej=pmb->je;
  else              sj=pmb->js,          ej=pmb->js+NGHOST-1;
  if(nb.ox3==0)     sk=pmb->ks,          ek=pmb->ke;
  else if(nb.ox3>0) sk=pmb->ke-NGHOST+1, ek=pmb->ke;
  else              sk=pmb->ks,          ek=pmb->ks+NGHOST-1;
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if(pmb->pmy_mesh->multilevel==true && nb.type != neighbor_face) {
    if(nb.ox1>0) ei++;
    if(nb.ox1<0) si--;
  }
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        buf[p++]=src.x1f(k,j,i);
    }
  }
  // bx2
  if(nb.ox1==0)      si=pmb->is,          ei=pmb->ie;
  else if(nb.ox1>0)  si=pmb->ie-NGHOST+1, ei=pmb->ie;
  else               si=pmb->is,          ei=pmb->is+NGHOST-1;
  if(pmb->block_size.nx2==1) sj=pmb->js,  ej=pmb->je;
  else if(nb.ox2==0) sj=pmb->js,          ej=pmb->je+1;
  else if(nb.ox2>0)  sj=pmb->je-NGHOST+1, ej=pmb->je;
  else               sj=pmb->js+1,        ej=pmb->js+NGHOST;
  if(pmb->pmy_mesh->multilevel==true && nb.type != neighbor_face) {
    if(nb.ox2>0) ej++;
    else if(nb.ox2<0) sj--;
  }
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        buf[p++]=src.x2f(k,j,i);
    }
  }
  // bx3
  if(nb.ox2==0)      sj=pmb->js,          ej=pmb->je;
  else if(nb.ox2>0)  sj=pmb->je-NGHOST+1, ej=pmb->je;
  else               sj=pmb->js,          ej=pmb->js+NGHOST-1;
  if(pmb->block_size.nx3==1) sk=pmb->ks,  ek=pmb->ke;
  else if(nb.ox3==0) sk=pmb->ks,          ek=pmb->ke+1;
  else if(nb.ox3>0)  sk=pmb->ke-NGHOST+1, ek=pmb->ke;
  else               sk=pmb->ks+1,        ek=pmb->ks+NGHOST;
  if(pmb->pmy_mesh->multilevel==true && nb.type != neighbor_face) {
    if(nb.ox3>0) ek++;
    else if(nb.ox3<0) sk--;
  }
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        buf[p++]=src.x3f(k,j,i);
    }
  }

  return p;
}

//--------------------------------------------------------------------------------------
//! \fn int BoundaryValues::LoadFieldBoundaryBufferToCoarser(InterfaceField &src,
//                                                 Real *buf, NeighborBlock& nb)
//  \brief Set field boundary buffers for sending to a block on the coarser level
int BoundaryValues::LoadFieldBoundaryBufferToCoarser(InterfaceField &src, Real *buf,
                                                     NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;
  int si, sj, sk, ei, ej, ek;
  int cn=pmb->cnghost-1;
  int p=0;

  // restrict the data before sending
  si=(nb.ox1>0)?(pmb->cie-cn):pmb->cis;
  ei=(nb.ox1<0)?(pmb->cis+cn):pmb->cie;
  sj=(nb.ox2>0)?(pmb->cje-cn):pmb->cjs;
  ej=(nb.ox2<0)?(pmb->cjs+cn):pmb->cje;
  sk=(nb.ox3>0)?(pmb->cke-cn):pmb->cks;
  ek=(nb.ox3<0)?(pmb->cks+cn):pmb->cke;
  RestrictFieldX1(src, si, ei, sj, ej, sk, ek);
  for (int k=sk; k<=ek; k++) {
    for (int j=sj; j<=ej; j++) {
#pragma simd
      for (int i=si; i<=ei; i++)
          buf[p++]=coarse_b_.x1f(k,j,i);
    }
  }

  si=(nb.ox1>0)?(pmb->cie-cn):pmb->cis;
  ei=(nb.ox1<0)?(pmb->cis+cn):pmb->cie;
  sj=(nb.ox2>0)?(pmb->cje-cn):pmb->cjs;
  ej=(nb.ox2<0)?(pmb->cjs+cn):pmb->cje;
  sk=(nb.ox3>0)?(pmb->cke-cn):pmb->cks;
  ek=(nb.ox3<0)?(pmb->cks+cn):pmb->cke;
  RestrictFieldX2(src, si, ei, sj, ej, sk, ek);
  for (int k=sk; k<=ek; k++) {
    for (int j=sj; j<=ej; j++) {
#pragma simd
      for (int i=si; i<=ei; i++)
          buf[p++]=coarse_b_.x2f(k,j,i);
    }
  }

  si=(nb.ox1>0)?(pmb->cie-cn):pmb->cis;
  ei=(nb.ox1<0)?(pmb->cis+cn):pmb->cie;
  sj=(nb.ox2>0)?(pmb->cje-cn):pmb->cjs;
  ej=(nb.ox2<0)?(pmb->cjs+cn):pmb->cje;
  sk=(nb.ox3>0)?(pmb->cke-cn):pmb->cks;
  ek=(nb.ox3<0)?(pmb->cks+cn):pmb->cke;
  RestrictFieldX3(src, si, ei, sj, ej, sk, ek);
  for (int k=sk; k<=ek; k++) {
    for (int j=sj; j<=ej; j++) {
#pragma simd
      for (int i=si; i<=ei; i++)
          buf[p++]=coarse_b_.x3f(k,j,i);
    }
  }

  return p;
}

//--------------------------------------------------------------------------------------
//! \fn int BoundaryValues::LoadFieldBoundaryBufferToFiner(InterfaceField &src, 
//                                                 Real *buf, NeighborBlock& nb)
//  \brief Set field boundary buffers for sending to a block on the finer level
int BoundaryValues::LoadFieldBoundaryBufferToFiner(InterfaceField &src, Real *buf,
                                                   NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;
  int si, sj, sk, ei, ej, ek;
  int cn=pmb->cnghost-1;
  int p=0;

  // send the data first and later prolongate on the target block
  // need to add edges for faces, add corners for edges
  // bx1
  if(nb.ox1==0) {
    if(nb.fi1==1)   si=pmb->is+pmb->block_size.nx1/2-pmb->cnghost, ei=pmb->ie+1;
    else            si=pmb->is, ei=pmb->ie+1-pmb->block_size.nx1/2+pmb->cnghost;
  }
  else if(nb.ox1>0) si=pmb->ie-cn, ei=pmb->ie;
  else              si=pmb->is+1,  ei=pmb->is+cn+1;
  if(nb.ox2==0) {
    sj=pmb->js,    ej=pmb->je;
    if(pmb->block_size.nx2 > 1) {
      if(nb.ox1!=0) {
        if(nb.fi1==1) sj+=pmb->block_size.nx2/2-pmb->cnghost;
        else          ej-=pmb->block_size.nx2/2-pmb->cnghost;
      }
      else {
        if(nb.fi2==1) sj+=pmb->block_size.nx2/2-pmb->cnghost;
        else          ej-=pmb->block_size.nx2/2-pmb->cnghost;
      }
    }
  }
  else if(nb.ox2>0) sj=pmb->je-cn, ej=pmb->je;
  else              sj=pmb->js,    ej=pmb->js+cn;
  if(nb.ox3==0) {
    sk=pmb->ks,    ek=pmb->ke;
    if(pmb->block_size.nx3 > 1) {
      if(nb.ox1!=0 && nb.ox2!=0) {
        if(nb.fi1==1) sk+=pmb->block_size.nx3/2-pmb->cnghost;
        else          ek-=pmb->block_size.nx3/2-pmb->cnghost;
      }
      else {
        if(nb.fi2==1) sk+=pmb->block_size.nx3/2-pmb->cnghost;
        else          ek-=pmb->block_size.nx3/2-pmb->cnghost;
      }
    }
  }
  else if(nb.ox3>0) sk=pmb->ke-cn, ek=pmb->ke;
  else              sk=pmb->ks,    ek=pmb->ks+cn;
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        buf[p++]=src.x1f(k,j,i);
    }
  }

  // bx2
  if(nb.ox1==0) {
    if(nb.fi1==1)   si=pmb->is+pmb->block_size.nx1/2-pmb->cnghost, ei=pmb->ie;
    else            si=pmb->is, ei=pmb->ie-pmb->block_size.nx1/2+pmb->cnghost;
  }
  else if(nb.ox1>0) si=pmb->ie-cn, ei=pmb->ie;
  else              si=pmb->is,    ei=pmb->is+cn;
  if(nb.ox2==0) {
    sj=pmb->js,    ej=pmb->je;
    if(pmb->block_size.nx2 > 1) {
      ej++;
      if(nb.ox1!=0) {
        if(nb.fi1==1) sj+=pmb->block_size.nx2/2-pmb->cnghost;
        else          ej-=pmb->block_size.nx2/2-pmb->cnghost;
      }
      else {
        if(nb.fi2==1) sj+=pmb->block_size.nx2/2-pmb->cnghost;
        else          ej-=pmb->block_size.nx2/2-pmb->cnghost;
      }
    }
  }
  else if(nb.ox2>0) sj=pmb->je-cn, ej=pmb->je;
  else              sj=pmb->js+1,  ej=pmb->js+cn+1;
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        buf[p++]=src.x2f(k,j,i);
    }
  }

  // bx3
  if(nb.ox2==0) {
    sj=pmb->js,    ej=pmb->je;
    if(pmb->block_size.nx2 > 1) {
      if(nb.ox1!=0) {
        if(nb.fi1==1) sj+=pmb->block_size.nx2/2-pmb->cnghost;
        else          ej-=pmb->block_size.nx2/2-pmb->cnghost;
      }
      else {
        if(nb.fi2==1) sj+=pmb->block_size.nx2/2-pmb->cnghost;
        else          ej-=pmb->block_size.nx2/2-pmb->cnghost;
      }
    }
  }
  else if(nb.ox2>0) sj=pmb->je-cn, ej=pmb->je;
  else              sj=pmb->js,    ej=pmb->js+cn;
  if(nb.ox3==0) {
    sk=pmb->ks,    ek=pmb->ke;
    if(pmb->block_size.nx3 > 1) {
      ek++;
      if(nb.ox1!=0 && nb.ox2!=0) {
        if(nb.fi1==1) sk+=pmb->block_size.nx3/2-pmb->cnghost;
        else          ek-=pmb->block_size.nx3/2-pmb->cnghost;
      }
      else {
        if(nb.fi2==1) sk+=pmb->block_size.nx3/2-pmb->cnghost;
        else          ek-=pmb->block_size.nx3/2-pmb->cnghost;
      }
    }
  }
  else if(nb.ox3>0) sk=pmb->ke-cn, ek=pmb->ke;
  else              sk=pmb->ks+1,  ek=pmb->ks+cn+1;
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        buf[p++]=src.x3f(k,j,i);
    }
  }

  return p;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SendFieldBoundaryBuffers(InterfaceField &src, int step)
//  \brief Send field boundary buffers
void BoundaryValues::SendFieldBoundaryBuffers(InterfaceField &src, int step)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb=pmb->neighbor[n];
    int ssize;
    if(nb.level==mylevel)
      ssize=LoadFieldBoundaryBufferSameLevel(src, field_send_[step][nb.bufid],nb);
    else if(nb.level<mylevel)
      ssize=LoadFieldBoundaryBufferToCoarser(src, field_send_[step][nb.bufid],nb);
    else
      ssize=LoadFieldBoundaryBufferToFiner(src, field_send_[step][nb.bufid], nb);
    if(nb.rank == myrank) { // on the same process
      MeshBlock *pbl=pmb->pmy_mesh->FindMeshBlock(nb.gid);
      // find target buffer
      std::memcpy(pbl->pbval->field_recv_[step][nb.targetid],
                  field_send_[step][nb.bufid], ssize*sizeof(Real));
      pbl->pbval->field_flag_[step][nb.targetid]=boundary_arrived;
    }
#ifdef MPI_PARALLEL
    else // MPI
      MPI_Start(&req_field_send_[step][nb.bufid]);
#endif
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SetFieldBoundarySameLevel(InterfaceField &dst,
//                                                     Real *buf, NeighborBlock& nb)
//  \brief Set field boundary received from a block on the same level
void BoundaryValues::SetFieldBoundarySameLevel(InterfaceField &dst, Real *buf,
                                               NeighborBlock& nb)
{
  MeshBlock *pmb=pmy_mblock_;
  int si, sj, sk, ei, ej, ek;

  int p=0;
  // bx1
  // for uniform grid: face-neighbors take care of the overlapping faces
  if(nb.ox1==0)     si=pmb->is,        ei=pmb->ie+1;
  else if(nb.ox1>0) si=pmb->ie+2,      ei=pmb->ie+NGHOST+1;
  else              si=pmb->is-NGHOST, ei=pmb->is-1;
  if(nb.ox2==0)     sj=pmb->js,        ej=pmb->je;
  else if(nb.ox2>0) sj=pmb->je+1,      ej=pmb->je+NGHOST;
  else              sj=pmb->js-NGHOST, ej=pmb->js-1;
  if(nb.ox3==0)     sk=pmb->ks,        ek=pmb->ke;
  else if(nb.ox3>0) sk=pmb->ke+1,      ek=pmb->ke+NGHOST;
  else              sk=pmb->ks-NGHOST, ek=pmb->ks-1;
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if(pmb->pmy_mesh->multilevel==true && nb.type != neighbor_face) {
    if(nb.ox1>0) si--;
    else if(nb.ox1<0) ei++;
  }
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        dst.x1f(k,j,i)=buf[p++];
    }
  }
  // bx2
  if(nb.ox1==0)      si=pmb->is,         ei=pmb->ie;
  else if(nb.ox1>0)  si=pmb->ie+1,       ei=pmb->ie+NGHOST;
  else               si=pmb->is-NGHOST,  ei=pmb->is-1;
  if(pmb->block_size.nx2==1) sj=pmb->js, ej=pmb->je;
  else if(nb.ox2==0) sj=pmb->js,         ej=pmb->je+1;
  else if(nb.ox2>0)  sj=pmb->je+2,       ej=pmb->je+NGHOST+1;
  else               sj=pmb->js-NGHOST,  ej=pmb->js-1;
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if(pmb->pmy_mesh->multilevel==true && nb.type != neighbor_face) {
    if(nb.ox2>0) sj--;
    else if(nb.ox2<0) ej++;
  }
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        dst.x2f(k,j,i)=buf[p++];
    }
  }
  if(pmb->block_size.nx2==1) { // 1D
#pragma simd
    for (int i=si; i<=ei; ++i)
      dst.x2f(sk,sj+1,i)=dst.x2f(sk,sj,i);
  }
  // bx3
  if(nb.ox2==0)      sj=pmb->js,         ej=pmb->je;
  else if(nb.ox2>0)  sj=pmb->je+1,       ej=pmb->je+NGHOST;
  else               sj=pmb->js-NGHOST,  ej=pmb->js-1;
  if(pmb->block_size.nx3==1) sk=pmb->ks, ek=pmb->ke;
  else if(nb.ox3==0) sk=pmb->ks,         ek=pmb->ke+1;
  else if(nb.ox3>0)  sk=pmb->ke+2,       ek=pmb->ke+NGHOST+1;
  else               sk=pmb->ks-NGHOST,  ek=pmb->ks-1;
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if(pmb->pmy_mesh->multilevel==true && nb.type != neighbor_face) {
    if(nb.ox3>0) sk--;
    else if(nb.ox3<0) ek++;
  }
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        dst.x3f(k,j,i)=buf[p++];
    }
  }
  if(pmb->block_size.nx3==1) { // 1D or 2D
    for (int j=sj; j<=ej; ++j) {
#pragma simd
      for (int i=si; i<=ei; ++i)
        dst.x3f(sk+1,j,i)=dst.x3f(sk,j,i);
    }
  }

  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SetFieldBoundaryFromCoarser(Real *buf, NeighborBlock& nb)
//  \brief Set field prolongation buffer received from a block on the same level
void BoundaryValues::SetFieldBoundaryFromCoarser(Real *buf, NeighborBlock& nb)
{
  int si, sj, sk, ei, ej, ek, ll;
  long int lx1, lx2, lx3;
  pmb->uid.GetLocation(lx1,lx2,lx3,ll);
  int cng=pmb->cnghost;
  int p=0;

  // bx1
  if(nb.ox1==0) {
    si=pmb->cis, ei=pmb->cie+1;
    if((lx1&1L)==0L) ei+=cng;
    else             si-=cng; 
  }
  else if(nb.ox1>0)  si=pmb->cie+2,   ei=pmb->cie+cng+1;
  else               si=pmb->cis-cng, ei=pmb->cis-1;
  if(nb.ox2==0) {
    sj=pmb->cjs, ej=pmb->cje;
    if(pmb->block_size.nx2 > 1) {
      if((lx2&1L)==0L) ej+=cng;
      else             sj-=cng; 
    }
  }
  else if(nb.ox2>0)  sj=pmb->cje+1,   ej=pmb->cje+cng;
  else               sj=pmb->cjs-cng, ej=pmb->cjs-1;
  if(nb.ox3==0) {
    sk=pmb->cks, ek=pmb->cke;
    if(pmb->block_size.nx3 > 1) {
      if((lx3&1L)==0L) ek+=cng;
      else             sk-=cng; 
    }
  }
  else if(nb.ox3>0)  sk=pmb->cke+1,   ek=pmb->cke+cng;
  else               sk=pmb->cks-cng, ek=pmb->cks-1;

  for (int n=0; n<(NFLUID); ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma simd
        for (int i=si; i<=ei; ++i)
          coarse_b_.x1f(k,j,i) = buf[p++];
      }
    }
  }

  // bx2
  if(nb.ox1==0) {
    si=pmb->cis, ei=pmb->cie+1;
    if((lx1&1L)==0L) ei+=cng;
    else             si-=cng; 
  }
  else if(nb.ox1>0)  si=pmb->cie+1,   ei=pmb->cie+cng;
  else               si=pmb->cis-cng, ei=pmb->cis-1;
  if(nb.ox2==0) {
    sj=pmb->cjs, ej=pmb->cje;
    if(pmb->block_size.nx2 > 1) {
      ej++;
      if((lx2&1L)==0L) ej+=cng;
      else             sj-=cng; 
    }
  }
  else if(nb.ox2>0)  sj=pmb->cje+2,   ej=pmb->cje+cng+1;
  else               sj=pmb->cjs-cng, ej=pmb->cjs-1;

  for (int n=0; n<(NFLUID); ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma simd
        for (int i=si; i<=ei; ++i)
          coarse_b_.x2f(k,j,i) = buf[p++];
      }
    }
  }

  // bx3
  if(nb.ox2==0) {
    sj=pmb->cjs, ej=pmb->cje;
    if(pmb->block_size.nx2 > 1) {
      if((lx2&1L)==0L) ej+=cng;
      else             sj-=cng; 
    }
  }
  else if(nb.ox2>0)  sj=pmb->cje+1,   ej=pmb->cje+cng;
  else               sj=pmb->cjs-cng, ej=pmb->cjs-1;
  if(nb.ox3==0) {
    sk=pmb->cks, ek=pmb->cke;
    if(pmb->block_size.nx3 > 1) {
      ek++;
      if((lx3&1L)==0L) ek+=cng;
      else             sk-=cng; 
    }
  }
  else if(nb.ox3>0)  sk=pmb->cke+2,   ek=pmb->cke+cng+1;
  else               sk=pmb->cks-cng, ek=pmb->cks-1;

  for (int n=0; n<(NFLUID); ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma simd
        for (int i=si; i<=ei; ++i)
          coarse_b_.x3f(k,j,i) = buf[p++];
      }
    }
  }

  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SetFielBoundaryFromFiner(InterfaceField &dst,
//                                                     Real *buf, NeighborBlock& nb)
//  \brief Set field boundary received from a block on the same level
void BoundaryValues::SetFieldBoundaryFromFiner(InterfaceField &dst, Real *buf,
                                               NeighborBlock& nb)
{
  return;
}


//--------------------------------------------------------------------------------------
//! \fn bool BoundaryValues::ReceiveFieldBoundaryBuffers(InterfaceField &dst, int step)
//  \brief load boundary buffer for x1 direction into the array
bool BoundaryValues::ReceiveFieldBoundaryBuffers(InterfaceField &dst, int step)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();
  int nc=0;

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb= pmb->neighbor[n];
    if(field_flag_[step][nb.bufid]==boundary_completed) { nc++; continue;}
    if(field_flag_[step][nb.bufid]==boundary_waiting) {
      if(nb.rank==myrank) // on the same process
        continue;
#ifdef MPI_PARALLEL
      else { // MPI boundary
        int test;
        MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&test,MPI_STATUS_IGNORE);
        MPI_Test(&req_field_recv_[step][nb.bufid],&test,MPI_STATUS_IGNORE);
        if(test==false) continue;
        field_flag_[step][nb.bufid] = boundary_arrived;
      }
#endif
    }
    if(nb.level==mylevel)
      SetFieldBoundarySameLevel(dst, field_recv_[step][nb.bufid], nb);
    else if(nb.level<mylevel)
      SetFieldBoundaryFromCoarser(field_recv_[step][nb.bufid], nb);
    else
      SetFieldBoundaryFromFiner(dst, field_recv_[step][nb.bufid], nb);

    field_flag_[step][nb.bufid] = boundary_completed; // completed
    nc++;
  }

  if(nc<pmb->nneighbor)
    return false;
  return true;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ReceiveFieldBoundaryBuffersWithWait(InterfaceField &dst,
//                                                               int step)
//  \brief load boundary buffer for x1 direction into the array
void BoundaryValues::ReceiveFieldBoundaryBuffersWithWait(InterfaceField &dst, int step)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb= pmb->neighbor[n];
#ifdef MPI_PARALLEL
    if(nb.rank!=myrank)
      MPI_Wait(&req_field_recv_[0][nb.bufid],MPI_STATUS_IGNORE);
#endif
    if(nb.level==mylevel)
      SetFieldBoundarySameLevel(dst, field_recv_[0][nb.bufid], nb);
    else if(nb.level<mylevel)
      SetFieldBoundaryFromCoarser(field_recv_[0][nb.bufid], nb);
    else
      SetFieldBoundaryFromFiner(dst, field_recv_[0][nb.bufid], nb);
    field_flag_[0][nb.bufid] = boundary_completed; // completed
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SendEMFCorrection(int step)
//  \brief Restrict, pack and send the surace EMF to the coarse neighbor(s) if needed
void BoundaryValues::SendEMFCorrection(int step)
{
  MeshBlock *pmb=pmy_mblock_;
  long int lx1, lx2, lx3;
  int mylevel;
  pmb->uid.GetLocation(lx1,lx2,lx3,mylevel);
  int fx1=lx1&1L, fx2=lx2&1L, fx3=lx3&1L;
  int fi1, fi2;
  AthenaArray<Real> &e1=pmb->pfield->e.x1e;
  AthenaArray<Real> &e2=pmb->pfield->e.x2e;
  AthenaArray<Real> &e3=pmb->pfield->e.x3e;
  AthenaArray<Real> &le1=sarea_[0];
  AthenaArray<Real> &le2=sarea_[1];

  for(int n=0; n<pmb->nneighbor; n++) {
    if(nb.type==neighbor_face) {
      if(nb.level==mylevel-1) {
        int p=0;
        if(pmb->block_size.nx3 > 1) { // 3D
          // x1 direction
          if(nb.fid==inner_x1 || nb.fid==outer_x1) {
            fi1=fx2, fi2=fx3;
            int i;
            if(nb.fid==inner_x1) i=pmb->is;
            else i=pmb->ie+1;
            // restrict and pack e2
            for(int k=pmb->ks; k<=pmb->ke+1; k+=2) {
              for(int j=pmb->js; j<=pmb->je; j+=2) {
                Real el1=pmb->pcoord->GetEdge2Length(k,j,i);
                Real el2=pmb->pcoord->GetEdge2Length(k,j+1,i);
                emfcor_fsend_[step][nb.fid][p++]=(e2(k,j,i)*el1+e2(k,j+1,i)*el2)/(el1+el2);
              }
            }
            // restrict and pack e3
            for(int k=pmb->ks; k<=pmb->ke; k+=2) {
              for(int j=pmb->js; j<=pmb->je+1; j+=2) {
                Real el1=pmb->pcoord->GetEdge3Length(k,j,i);
                Real el2=pmb->pcoord->GetEdge3Length(k+1,j,i);
                emfcor_fsend_[step][nb.fid][p++]=(e3(k,j,i)*el1+e3(k+1,j,i)*el2)/(el1+el2);
              }
            }
          }
          // x2 direction
          else if(nb.fid==inner_x2 || nb.fid==outer_x2) {
            fi1=fx1, fi2=fx3;
            int j;
            if(nb.fid==inner_x2) j=pmb->js;
            else j=pmb->je+1;
            // restrict and pack e1
            for(int k=pmb->ks; k<=pmb->ke+1; k+=2) {
              pmb->pcoord->Edge1Length(k, j, pmb->is, pmb->ie, le1);
              for(int i=pmb->is; i<=pmb->ie; i+=2)
                emfcor_fsend_[step][nb.fid][p++]
                            =(e1(k,j,i)*le1(i)+e1(k,j,i+1)*le1(i+1))/(le(i)+le1(i+1));
            }
            // restrict and pack e3
            for(int k=pmb->ks; k<=pmb->ke; k+=2) {
              pmb->pcoord->Edge3Length(k,   j, pmb->is, pmb->ie+1, le1);
              pmb->pcoord->Edge3Length(k+1, j, pmb->is, pmb->ie+1, le2);
              for(int i=pmb->is; i<=pmb->ie+1; i+=2)
                emfcor_fsend_[step][nb.fid][p++]
                            =(e3(k,j,i)*le1(i)+e3(k+1,j,i)*le2(i))/(le1(i)+le2(i));
            }
          }
          // x3 direction
          else if(nb.fid==inner_x3 || nb.fid==outer_x3) {
            fi1=fx1, fi2=fx2;
            int k;
            if(nb.fid==inner_x3) k=pmb->ks;
            else k=pmb->ke+1;
            // restrict and pacj e1
            for(int j=pmb->js; j<=pmb->je+1; j+=2) {
              pmb->pcoord->Edge1Length(k, j, pmb->is, pmb->ie, le1);
              for(int i=pmb->is; i<=pmb->ie; i+=2)
                emfcor_fsend_[step][nb.fid][p++]
                            =(e1(k,j,i)*le1(i)+e1(k,j,i+1)*le1(i+1))/(le(i)+le1(i+1));
            }
            // restrict and pack e2
            for(int j=pmb->js; j<=pmb->je; j+=2) {
              pmb->pcoord->Edge2Length(k,   j, pmb->is, pmb->ie+1, le1);
              pmb->pcoord->Edge2Length(k, j+1, pmb->is, pmb->ie+1, le2);
              for(int i=pmb->is; i<=pmb->ie+1; i+=2)
                emfcor_fsend_[step][nb.fid][p++]
                            =(e2(k,j,i)*le1(i)+e2(k,j+1,i)*le2(i))/(le1(i)+le2(i));
            }
          }
        }
        else if(pmb->block_size.nx2 > 1) { // 2D
          int k=pmb->ks;
          // x1 direction
          if(nb.fid==inner_x1 || nb.fid==outer_x1) {
            fi1=fx2, fi2=fx3;
            int i;
            if(nb.fid==inner_x1) i=pmb->is;
            else i=pmb->ie+1;
            // restrict and pack e2
            for(int j=pmb->js; j<=pmb->je; j+=2) {
              Real el1=pmb->pcoord->GetEdge2Length(k,j,i);
              Real el2=pmb->pcoord->GetEdge2Length(k,j+1,i);
              emfcor_fsend_[step][nb.fid][p++]=(e2(k,j,i)*el1+e2(k,j+1,i)*el2)/(el1+el2);
            }
            // pack e3
            for(int j=pmb->js; j<=pmb->je+1; j+=2)
              emfcor_fsend_[step][nb.fid][p++]=e3(k,j,i);
          }
          // x2 direction
          else if(nb.fid==inner_x2 || nb.fid==outer_x2) {
            fi1=fx1, fi2=fx3;
            int j;
            if(nb.fid==inner_x2) j=pmb->js;
            else j=pmb->je+1;
            // restrict and pack e1
            pmb->pcoord->Edge1Length(k, j, pmb->is, pmb->ie, le1);
            for(int i=pmb->is; i<=pmb->ie; i+=2) {
              emfcor_fsend_[step][nb.fid][p++]
                          =(e1(k,j,i)*le1(i)+e1(k,j,i+1)*le1(i+1))/(le(i)+le1(i+1));
            }
            // pack e3
            for(int i=pmb->is; i<=pmb->ie+1; i+=2)
              emfcor_fsend_[step][nb.fid][p++]=e3(k,j,i);
          }
        }
        else { // 1D
          fi1=fx2, fi2=fx3;
          int i, j=pmb->js, k=pmb->ks;
          if(nb.fid==inner_x1) i=is;
          else i=ie+1;
          // pack e2 and e3
          emfcor_fsend_[step][nb.fid][p++]=e2(k,j,i);
          emfcor_fsend_[step][nb.fid][p++]=e3(k,j,i);
        }
        if(nb.rank==myrank) { // on the same node
          MeshBlock *pbl=pmb->pmy_mesh->FindMeshBlock(nb.gid);
          std::memcpy(pbl->pbval->emfcor_frecv_[step][(nb.fid^1)][fi2][fi1],
                      emfcor_fsend_[step][nb.fid], p*sizeof(Real));
          pbl->pbval->emfcor_fflag_[step][(nb.fid^1)][fi2][fi1]=boundary_arrived;
        }
#ifdef MPI_PARALLEL
        else
          MPI_Start(&req_emfcor_fsend_[step][nb.fid]);
#endif
      }
    }
    else if(nb.type==neighbor_edge) {
      if(nb.level==mylevel-1 && edge_flag_[nb.eid]==true) {
        int p=0;
        if(pmb->block_size.nx3 > 1) { // 3D
          // x1x2 edge
          if(nb.eid>=0 && nb.eid<4) {
            fi1=fx3;
            int i, j;
            if((nb.eid&1)==0) i=pmb->is;
            else i=pmb->ie+1;
            if((nb.eid&2)==0) j=pmb->js;
            else j=pmb->je+1;
            // restrict and pack e3
            for(int k=pmb->ks; k<=pmb->ke; k+=2) {
              Real el1=pmb->pcoord->GetEdge3Length(k,j,i);
              Real el2=pmb->pcoord->GetEdge3Length(k+1,j,i);
              emfcor_esend_[step][nb.eid][p++]=(e3(k,j,i)*el1+e3(k+1,j,i)*el2)/(el1+el2);
            }
          }
          // x1x3 edge
          else if(nb.eid>=4 && nb.eid<8) {
            fi1=fx2;
            int i, k;
            if((nb.eid&1)==0) i=pmb->is;
            else i=pmb->ie+1;
            if((nb.eid&2)==0) k=pmb->ks;
            else k=pmb->ke+1;
            // restrict and pack e2
            for(int j=pmb->js; j<=pmb->je; j+=2) {
              Real el1=pmb->pcoord->GetEdge2Length(k,j,i);
              Real el2=pmb->pcoord->GetEdge2Length(k,j+1,i);
              emfcor_esend_[step][nb.eid][p++]=(e2(k,j,i)*el1+e2(k,j+1,i)*el2)/(el1+el2);
            }
          }
          // x2x3 edge
          else if(nb.eid>=8 && nb.eid<12) {
            fi1=fx1;
            int j, k;
            if((nb.eid&1)==0) j=pmb->js;
            else j=pmb->je+1;
            if((nb.eid&2)==0) k=pmb->ks;
            else k=pmb->ke+1;
            // restrict and pack e1
            pmb->pcoord->Edge1Length(k, j, pmb->is, pmb->ie, le1);
            for(int i=pmb->is; i<=pmb->ie; i+=2)
              emfcor_esend_[step][nb.eid][p++]
                          =(e1(k,j,i)*le1(i)+e1(k,j,i+1)*le1(i+1))/(le(i)+le1(i+1));
          }
        }
        else if(pmb->block_size.nx2 > 1) { // 2D
          // x1x2 edge
          fi1=fx3;
          int i, j;
          if((nb.eid&1)==0) i=pmb->is;
          else i=pmb->ie+1;
          if((nb.eid&2)==0) j=pmb->js;
          else j=pmb->je+1;
          // pack e3
          emfcor_esend_[step][nb.eid][p++]=e3(pmb->ks,j,i)
        }
        if(nb.rank==myrank) { // on the same node
          MeshBlock *pbl=pmb->pmy_mesh->FindMeshBlock(nb.gid);
          std::memcpy(pbl->pbval->emfcor_erecv_[step][(nb.eid^3)][fi1],
                      emfcor_esend_[step][nb.eid], p*sizeof(Real));
          pbl->pbval->emfcor_eflag_[step][(nb.eid^3)][fi1]=boundary_arrived;
        }
#ifdef MPI_PARALLEL
        else
          MPI_Start(&req_emfcor_esend_[step][nb.eid]);
#endif
      }
    }
    else break;
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ReceiveEMFCorrection(int step)
//  \brief Receive and Apply the surace EMF to the coarse neighbor(s) if needed
bool BoundaryValues::ReceiveEMFCorrection(int step)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();
  int nc=0, nff=0;

  // count the number of finer faces and edges
  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb= pmb->neighbor[n];
    if(nb.level==mylevel+1) {
      if(nb.type==neighbor_face) nff++;
      else if(nb.type==neighbor_edge) {
       if(edge_flag_[nb.eid]==true) nff++;
      }
      else break;
    }
  }

  for(int n=0; n<pmb->nneighbor; n++) {
    NeighborBlock& nb= pmb->neighbor[n];
    if(nb.level==mylevel+1) {
      if(nb.type==neighbor_face) {
        if(emfcor_fflag_[step][nb.fid][nb.fi2][nb.fi1]==boundary_completed) { nc++; continue; }
        if(emfcor_fflag_[step][nb.fid][nb.fi2][nb.fi1]==boundary_waiting) {
          if(nb.rank==myrank) // on the same process
            continue;
#ifdef MPI_PARALLEL
          else { // MPI boundary
            int test;
            MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&test,MPI_STATUS_IGNORE);
            MPI_Test(&req_emfcor_frecv_[step][nb.fid][nb.fi2][nb.fi1],&test,MPI_STATUS_IGNORE);
            if(test==false) continue;
            emfcor_fflag_[step][nb.fid][nb.fi2][nb.fi1] = boundary_arrived;
          }
#endif
        }
        // boundary arrived; apply EMF correction
        Real *buf=emfcor_frecv_[step][nb.fid][nb.fi2][nb.fi1];
        int p=0;
        if(pmb->block_size.nx3 > 1) { // 3D
          // x1 direction
          if(nb.fid==inner_x1 || nb.fid==inner_x2) {
            int i, jl=pmb->js, ju=pmb->je+1, kl=pmb->ks, ku=pmb->ke+1;
            if(nb.fid==inner_x1) i=pmb->is;
            else i=pmb->ie+1;
            if(nb.fi1==0) ju=pmb->js+pmb->block_size.nx2/2-1;
            else jl=pmb->js+pmb->block_size.nx2/2;
            if(nb.fi2==0) ku=pmb->ks+pmb->block_size.nx3/2-1;
            else kl=pmb->ks+pmb->block_size.nx3/2;
            // unpack e2
            for(int k=kl; k<=ku+1; k++) {
              for(int j=jl; j<=ju; j++)
                e2(k,j,i)=buf[p++];
            }
            // unpack e3
            for(int k=kl; k<=ku; k++) {
              for(int j=jl; j<=ju+1; j++)
                e3(k,j,i)=buf[p++];
            }
          }
          // x2 direction
          else if(nb.fid==inner_x2 || nb.fid==outer_x2) {
            int j, il=pmb->is, iu=pmb->ie+1, kl=pmb->ks, ku=pmb->ke+1;
            if(nb.fid==inner_x2) j=pmb->js;
            else j=pmb->je+1;
            if(nb.fi1==0) iu=pmb->is+pmb->block_size.nx1/2-1;
            else il=pmb->is+pmb->block_size.nx1/2;
            if(nb.fi2==0) ku=pmb->ks+pmb->block_size.nx3/2-1;
            else kl=pmb->ks+pmb->block_size.nx3/2;
            // unpack e1
            for(int k=kl; k<=ku+1; k++) {
              for(int i=il; i<=iu; i++)
                e1(k,j,i)=buf[step][nb.fid][p++];
            }
            // unpack e3
            for(int k=kl; k<=ku; k++) {
              for(int i=il; i<=iu+1; i++)
                e3(k,j,i)=buf[step][nb.fid][p++];
            }
          }
          // x3 direction
          else if(nb.fid==inner_x3 || nb.fid==outer_x3) {
            int k, il=pmb->is, iu=pmb->ie+1, jl=pmb->js, ju=pmb->je+1;
            if(nb.fid==inner_x3) k=pmb->ks;
            else k=pmb->ke+1;
            if(nb.fi1==0) iu=pmb->is+pmb->block_size.nx1/2-1;
            else il=pmb->is+pmb->block_size.nx1/2;
            if(nb.fi2==0) ju=pmb->js+pmb->block_size.nx2/2-1;
            else jl=pmb->js+pmb->block_size.nx2/2;
            // unpack e1
            for(int j=jl; j<=ju+1; j++) {
              for(int i=il; i<=iu; i++)
                e1(k,j,i)=buf[step][nb.fid][p++];
            }
            // unpack e2
            for(int j=jl; j<=ju; j++) {
              for(int i=il; i<=iu+1; i++)
                e3(k,j,i)=buf[step][nb.fid][p++];
            }
          }
        }
        else if(block_size.nx2 > 1) { // 2D
          int k=pmb->ks;
          // x1 direction
          if(nb.fid==inner_x1 || nb.fid==outer_x1) {
            int i, jl=pmb->js, ju=pmb->je+1;
            if(nb.fid==inner_x1) i=pmb->is;
            else i=pmb->ie+1;
            if(nb.fi1==0) ju=pmb->js+pmb->block_size.nx2/2-1;
            else jl=pmb->js+pmb->block_size.nx2/2;
            // unpack e2
            for(int j=jl; j<=ju; j++)
              e2(k+1,j,i)=e2(k,j,i)=buf[p++];
            // unpack e3
            for(int j=jl; j<=ju+1; j++)
              e3(k,j,i)=buf[p++];
          }
          // x2 direction
          else if(nb.fid==inner_x2 || nb.fid==outer_x2) {
            int j, il=pmb->is, iu=pmb->ie+1;
            if(nb.fid==inner_x2) j=pmb->js;
            else j=pmb->je+1;
            if(nb.fi1==0) iu=pmb->is+pmb->block_size.nx1/2-1;
            else il=pmb->is+pmb->block_size.nx1/2;
            // unpack e1
            for(int i=il; i<=iu; i++)
              e1(k+1,j,i)=e1(k,j,i)=buf[p++];
            // unpack e3
            for(int i=il; i<=iu+1; i++)
              e3(k,j,i)=buf[p++];
          }
        }
        else { // 1D
          int i, j=pmb->js, k=pmb->ks;
          if(nb.fid==inner_x1) i=il;
          else i=iu;
          // unpack e2
          e2(k+1,j,i)=e2(k,j,i)=buf[p++];
          // unpack e3
          e3(k,j+1,i)=e3(k,j,i)=buf[p++];
        }
        emfcor_fflag_[step][nb.fid][nb.fi2][nb.fi1] = boundary_completed;
        nc++;
      }
      else if(nb.type==neighbor_edge) {
        if(edge_flag_[nb.eid]!=true) continue;
        if(emfcor_eflag_[step][nb.eid][nb.fi1]==boundary_completed) { nc++; continue; }
        if(emfcor_eflag_[step][nb.eid][nb.fi1]==boundary_waiting) {
          if(nb.rank==myrank) // on the same process
            continue;
#ifdef MPI_PARALLEL
          else { // MPI boundary
            int test;
            MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&test,MPI_STATUS_IGNORE);
            MPI_Test(&req_emfcor_erecv_[step][nb.eid][nb.fi1],&test,MPI_STATUS_IGNORE);
            if(test==false) continue;
            emfcor_eflag_[step][nb.eid][nb.fi1] = boundary_arrived;
          }
#endif
        }
        // boundary arrived; apply EMF correction
        Real *buf=emfcor_erecv_[step][nb.eid][nb.fi1];
        int p=0;
        if(pmb->block_size.nx3 > 1) { // 3D
          // x1x2 edge
          if(nb.eid>=0 && nb.eid<4) {
            int i, j, kl=pmb->ks, ku=pmb->ke+1;
            if((nb.eid&1)==0) i=pmb->is;
            else i=pmb->ie+1;
            if((nb.eid&2)==0) j=pmb->js;
            else j=pmb->je+1;
            if(nb.fi1==0) ku=pmb->ks+pmb->block_size.nx3/2-1;
            else kl=pmb->ks+pmb->block_size.nx3/2;
            // unpack e3
            for(int k=kl; k<=ku; k++)
              e3(k,j,i)=buf[p++];
          }
          // x1x3 edge
          else if(nb.eid>=4 && nb.eid<8) {
            int i, k, jl=pmb->js, ju=pmb->je+1;
            if((nb.eid&1)==0) i=pmb->is;
            else i=pmb->ie+1;
            if((nb.eid&2)==0) k=pmb->ks;
            else k=pmb->ke+1;
            if(nb.fi1==0) ju=pmb->js+pmb->block_size.nx2/2-1;
            else jl=pmb->js+pmb->block_size.nx2/2;
            // unpack e2
            for(int j=jl; j<=ju; j++)
              e2(k,j,i)=buf[p++];
          }
          // x2x3 edge
          else if(nb.eid>=8 && nb.eid<12) {
            int j, k, il=pmb->is, iu=pmb->ie+1;
            if((nb.eid&1)==0) j=pmb->js;
            else j=pmb->je+1;
            if((nb.eid&2)==0) k=pmb->ks;
            else k=pmb->ke+1;
            if(nb.fi1==0) iu=pmb->is+pmb->block_size.nx1/2-1;
            else il=pmb->is+pmb->block_size.nx1/2;
            // unpack e1
            for(int i=pmb->is; i<=pmb->ie; i+=2)
              e1(k,j,i)=buf[p++;]
          }
        }
        else if(pmb->block_size.nx2 > 1) { // 2D
          int i, j, k=pmb->ks;
          if((nb.eid&1)==0) i=pmb->is;
          else i=pmb->ie+1;
          if((nb.eid&2)==0) j=pmb->js;
          else j=pmb->je+1;
          // unpack e3
          e3(k,j,i)=buf[p++];
        }
        emfcor_eflag_[step][nb.eid][nb.fi1] = boundary_completed;
        nc++;
      }
      else break;
    }
  }

  if(nc<nff)
    return false;
  return true;
}

//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ClearBoundaryForInit(void)
//  \brief clean up the boundary flags for initialization
void BoundaryValues::ClearBoundaryForInit(void)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();

  for(int n=0;n<pmb->nneighbor;n++) {
    NeighborBlock& nb=pmb->neighbor[n];
    fluid_flag_[0][nb.bufid] = boundary_waiting;
    if (MAGNETIC_FIELDS_ENABLED)
      field_flag_[0][nb.bufid] = boundary_waiting;
#ifdef MPI_PARALLEL
    if(nb.rank!=myrank) {
      MPI_Wait(&req_fluid_send_[0][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
      if (MAGNETIC_FIELDS_ENABLED)
        MPI_Wait(&req_field_send_[0][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
    }
#endif
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ClearBoundaryAll(void)
//  \brief clean up the boundary flags after each loop
void BoundaryValues::ClearBoundaryAll(void)
{
  MeshBlock *pmb=pmy_mblock_;
  int mylevel=pmb->uid.GetLevel();
  for(int l=0;l<NSTEP;l++) {
    for(int n=0;n<pmb->nneighbor;n++) {
      NeighborBlock& nb=pmb->neighbor[n];
      fluid_flag_[l][nb.bufid] = boundary_waiting;
      if(nb.type==neighbor_face)
        flcor_flag_[l][nb.fid][nb.fi2][nb.fi1] = boundary_waiting;
      if (MAGNETIC_FIELDS_ENABLED)
        field_flag_[l][nb.bufid] = boundary_waiting;
#ifdef MPI_PARALLEL
      if(nb.rank!=myrank) {
        MPI_Wait(&req_fluid_send_[l][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
        if(nb.type==neighbor_face && nb.level<mylevel)
          MPI_Wait(&req_flcor_send_[l][nb.fid],MPI_STATUS_IGNORE); // Wait for Isend
        if (MAGNETIC_FIELDS_ENABLED) {
          MPI_Wait(&req_field_send_[l][nb.bufid],MPI_STATUS_IGNORE); // Wait for Isend
          if(nb.type==neighbor_face && nb.level<mylevel)
            MPI_Wait(&req_emfcor_fsend_[l][nb.fid],MPI_STATUS_IGNORE); // Wait for Isend
          if(nb.type==neighbor_edge && nb.level<mylevel && edge_flag_[nb.eid]==true)
            MPI_Wait(&req_emfcor_esend_[l][nb.eid],MPI_STATUS_IGNORE); // Wait for Isend
        }
      }
#endif
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::FluidPhysicalBoundaries(AthenaArray<Real> &dst)
//  \brief Apply physical boundary conditions for fluid
void BoundaryValues::FluidPhysicalBoundaries(AthenaArray<Real> &dst)
{
  MeshBlock *pmb=pmy_mblock_;
  int bis=pmb->is, bie=pmb->ie, bjs=pmb->js, bje=pmb->je, bks=pmb->ks, bke=pmb->ke;

  if(pmb->pmy_mesh->face_only==false) { // extend the ghost zone
    bis=pmb->is-NGHOST;
    bie=pmb->ie+NGHOST;
    if(FluidBoundary_[inner_x2]==NULL && pmb->block_size.nx2>1) bjs=pmb->js-NGHOST;
    if(FluidBoundary_[outer_x2]==NULL && pmb->block_size.nx2>1) bje=pmb->je+NGHOST;
    if(FluidBoundary_[inner_x3]==NULL && pmb->block_size.nx3>1) bks=pmb->ks-NGHOST;
    if(FluidBoundary_[outer_x3]==NULL && pmb->block_size.nx3>1) bke=pmb->ke+NGHOST;
  }

  if(FluidBoundary_[inner_x1]!=NULL)
    FluidBoundary_[inner_x1](pmb, dst, pmb->is, pmb->ie, bjs, bje, bks, bke);
  if(FluidBoundary_[outer_x1]!=NULL)
    FluidBoundary_[outer_x1](pmb, dst, pmb->is, pmb->ie, bjs, bje, bks, bke);
  if(pmb->block_size.nx2>1) { // 2D or 3D
    if(FluidBoundary_[inner_x2]!=NULL)
      FluidBoundary_[inner_x2](pmb, dst, bis, bie, pmb->js, pmb->je, bks, bke);
    if(FluidBoundary_[outer_x2]!=NULL)
      FluidBoundary_[outer_x2](pmb, dst, bis, bie, pmb->js, pmb->je, bks, bke);
  }
  if(pmb->block_size.nx3>1) { // 3D
    if(pmb->pmy_mesh->face_only==false) {
      bjs=pmb->js-NGHOST;
      bje=pmb->je+NGHOST;
    }
    if(FluidBoundary_[inner_x3]!=NULL)
      FluidBoundary_[inner_x3](pmb, dst, bis, bie, bjs, bje, pmb->ks, pmb->ke);
    if(FluidBoundary_[outer_x3]!=NULL)
      FluidBoundary_[outer_x3](pmb, dst, bis, bie, bjs, bje, pmb->ks, pmb->ke);
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BoundaryValues::FieldPhysicalBoundaries(AthenaArray<Real> &dst)
//  \brief Apply physical boundary conditions for field
void BoundaryValues::FieldPhysicalBoundaries(InterfaceField &dst)
{
  MeshBlock *pmb=pmy_mblock_;
  int bis=pmb->is-NGHOST, bie=pmb->ie+NGHOST;
  int bjs=pmb->js, bje=pmb->je, bks=pmb->ks, bke=pmb->ke;

  if(FieldBoundary_[inner_x2]==NULL && pmb->block_size.nx2>1) bjs=pmb->js-NGHOST;
  if(FieldBoundary_[outer_x2]==NULL && pmb->block_size.nx2>1) bje=pmb->je+NGHOST;
  if(FieldBoundary_[inner_x3]==NULL && pmb->block_size.nx3>1) bks=pmb->ks-NGHOST;
  if(FieldBoundary_[outer_x3]==NULL && pmb->block_size.nx3>1) bke=pmb->ke+NGHOST;

  if(FieldBoundary_[inner_x1]!=NULL)
    FieldBoundary_[inner_x1](pmb, dst, pmb->is, pmb->ie, bjs, bje, bks, bke);
  if(FieldBoundary_[outer_x1]!=NULL)
    FieldBoundary_[outer_x1](pmb, dst, pmb->is, pmb->ie, bjs, bje, bks, bke);
  if(pmb->block_size.nx2>1) { // 2D or 3D
    if(FieldBoundary_[inner_x2]!=NULL)
      FieldBoundary_[inner_x2](pmb, dst, bis, bie, pmb->js, pmb->je, bks, bke);
    if(FieldBoundary_[outer_x2]!=NULL)
      FieldBoundary_[outer_x2](pmb, dst, bis, bie, pmb->js, pmb->je, bks, bke);
  }
  if(pmb->block_size.nx3>1) { // 3D
    bjs=pmb->js-NGHOST;
    bje=pmb->je+NGHOST;
    if(FieldBoundary_[inner_x3]!=NULL)
      FieldBoundary_[inner_x3](pmb, dst, bis, bie, bjs, bje, pmb->ks, pmb->ke);
    if(FieldBoundary_[outer_x3]!=NULL)
      FieldBoundary_[outer_x3](pmb, dst, bis, bie, bjs, bje, pmb->ks, pmb->ke);
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn unsigned int CreateBufferID(int ox1, int ox2, int ox3, int fi1, int fi2)
//  \brief calculate a buffer identifier
unsigned int CreateBufferID(int ox1, int ox2, int ox3, int fi1, int fi2)
{
  unsigned int ux1=(unsigned)(ox1+1);
  unsigned int ux2=(unsigned)(ox2+1);
  unsigned int ux3=(unsigned)(ox3+1);
  return (ux1<<6) | (ux2<<4) | (ux3<<2) | (fi1<<1) | fi2;
}


//--------------------------------------------------------------------------------------
//! \fn unsigned int CreateMPITag(int lid, int flag, int phys, int bufid)
//  \brief calculate an MPI tag
unsigned int CreateMPITag(int lid, int flag, int phys, int bufid)
{
// tag = local id of destination (18) + flag (2) + physics (4) + bufid(7)
  return (lid<<13) | (flag<<11) | (phys<<7) | bufid;
}


//--------------------------------------------------------------------------------------
//! \fn int BufferID(int dim, bool multilevel, bool face_only)
//  \brief calculate neighbor indexes and target buffer IDs
int BufferID(int dim, bool multilevel, bool face_only)
{
  int nf1=1, nf2=1;
  if(multilevel==true) {
    if(dim>=2) nf1=2;
    if(dim>=3) nf2=2;
  }
  int b=0;
  // x1 face
  for(int n=-1; n<=1; n+=2) {
    for(int f2=0;f2<nf2;f2++) {
      for(int f1=0;f1<nf1;f1++) {
        ni_[b].ox1=n; ni_[b].ox2=0; ni_[b].ox3=0;
        ni_[b].fi1=f1; ni_[b].fi2=f2; ni_[b].type=neighbor_face;
        b++;
      }
    }
  }
  // x2 face
  if(dim>=2) {
    for(int n=-1; n<=1; n+=2) {
      for(int f2=0;f2<nf2;f2++) {
        for(int f1=0;f1<nf1;f1++) {
          ni_[b].ox1=0; ni_[b].ox2=n; ni_[b].ox3=0;
          ni_[b].fi1=f1; ni_[b].fi2=f2; ni_[b].type=neighbor_face;
          b++;
        }
      }
    }
  }
  if(dim==3) {
    // x3 face
    for(int n=-1; n<=1; n+=2) {
      for(int f2=0;f2<nf2;f2++) {
        for(int f1=0;f1<nf1;f1++) {
          ni_[b].ox1=0; ni_[b].ox2=0; ni_[b].ox3=n;
          ni_[b].fi1=f1; ni_[b].fi2=f2; ni_[b].type=neighbor_face;
          b++;
        }
      }
    }
  }
  // edges
  // x1x2
  if(dim>=2) {
    for(int m=-1; m<=1; m+=2) {
      for(int n=-1; n<=1; n+=2) {
        for(int f1=0;f1<nf1;f1++) {
          ni_[b].ox1=n; ni_[b].ox2=m; ni_[b].ox3=0;
          ni_[b].fi1=f1; ni_[b].fi2=0; ni_[b].type=neighbor_edge;
          b++;
        }
      }
    }
  }
  if(dim==3) {
    // x1x3
    for(int m=-1; m<=1; m+=2) {
      for(int n=-1; n<=1; n+=2) {
        for(int f1=0;f1<nf1;f1++) {
          ni_[b].ox1=n; ni_[b].ox2=0; ni_[b].ox3=m;
          ni_[b].fi1=f1; ni_[b].fi2=0; ni_[b].type=neighbor_edge;
          b++;
        }
      }
    }
    // x2x3
    for(int m=-1; m<=1; m+=2) {
      for(int n=-1; n<=1; n+=2) {
        for(int f1=0;f1<nf1;f1++) {
          ni_[b].ox1=0; ni_[b].ox2=n; ni_[b].ox3=m;
          ni_[b].fi1=f1; ni_[b].fi2=0; ni_[b].type=neighbor_edge;
          b++;
        }
      }
    }
    // corners
    for(int l=-1; l<=1; l+=2) {
      for(int m=-1; m<=1; m+=2) {
        for(int n=-1; n<=1; n+=2) {
          ni_[b].ox1=n; ni_[b].ox2=m; ni_[b].ox3=l;
          ni_[b].fi1=0; ni_[b].fi2=0; ni_[b].type=neighbor_corner;
          b++;
        }
      }
    }
  }

  for(int n=0;n<b;n++)
    bufid_[n]=CreateBufferID(ni_[n].ox1, ni_[n].ox2, ni_[n].ox3, ni_[n].fi1, ni_[n].fi2);

  return b;
}

int FindBufferID(int ox1, int ox2, int ox3, int fi1, int fi2, int bmax)
{
  int bid=CreateBufferID(ox1, ox2, ox3, fi1, fi2);

  for(int i=0;i<bmax;i++) {
    if(bid==bufid_[i]) return i;
  }
  return -1;
}

