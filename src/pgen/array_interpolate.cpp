//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file array_interpolate.cpp
//! \brief Problem generator for initializing with preexisting array from HDF5 input with a grid remapping
//! based on from_array.cpp

// C headers

// C++ headers
#include <algorithm>  // max()
#include <string>     // c_str(), string

// Athena++ headers
#include "../athena.hpp"              // Real
#include "../athena_arrays.hpp"       // AthenaArray
#include "../field/field.hpp"         // Field
#include "../globals.hpp"             // Globals
#include "../hydro/hydro.hpp"         // Hydro
#include "../inputs/hdf5_reader.hpp"  // HDF5ReadRealArray()
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"     // ParameterInput


//----------------------------------------------------------------------------------------
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function for setting initial conditions
//!
//! Inputs:
//! - pin: parameters
//! Outputs: (none)
//! Notes:
//! - uses input parameters to determine which file contains array of conserved values
//!   dataset must be 5-dimensional array with the following sizes:
//!   - NHYDRO
//!   - total number of MeshBlocks
//!   - MeshBlock/nx3
//!   - MeshBlock/nx2
//!   - MeshBlock/nx1

namespace{
    Real bgdrho = 1e-8, bgdp = 1e-8, rotangle = 0.;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
    bgdrho = pin->GetReal("problem","bgdrho");
    bgdp = pin->GetReal("problem","bgdp");
    rotangle = pin->GetOrAddReal("problem","rotangle", 0.);
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("problem","four_pi_G");
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);
  }
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Determine locations of initial values
  std::string input_filename = pin->GetString("problem", "input_filename");
  std::string dataset_cons = pin->GetString("problem", "dataset_cons");
  int index_dens = pin->GetInteger("problem", "index_dens");
  int index_mom1 = pin->GetInteger("problem", "index_mom1");
  int index_mom2 = pin->GetInteger("problem", "index_mom2");
  int index_mom3 = pin->GetInteger("problem", "index_mom3");
  int index_etot = pin->GetInteger("problem", "index_etot");
  std::string dataset_b1 = pin->GetString("problem", "dataset_b1");
  std::string dataset_b2 = pin->GetString("problem", "dataset_b2");
  std::string dataset_b3 = pin->GetString("problem", "dataset_b3");

  // Set conserved array selections
  int start_cons_file[5];
  start_cons_file[1] = gid;
  start_cons_file[2] = 0;
  start_cons_file[3] = 0;
  start_cons_file[4] = 0;
  int start_cons_indices[5];
  start_cons_indices[IDN] = index_dens;
  start_cons_indices[IM1] = index_mom1;
  start_cons_indices[IM2] = index_mom2;
  start_cons_indices[IM3] = index_mom3;
  start_cons_indices[IEN] = index_etot;
  int count_cons_file[5];
  count_cons_file[0] = 1;
  count_cons_file[1] = 1;
  count_cons_file[2] = block_size.nx3;
  count_cons_file[3] = block_size.nx2;
  count_cons_file[4] = block_size.nx1;
  int start_cons_mem[4];
  start_cons_mem[1] = ks;
  start_cons_mem[2] = js;
  start_cons_mem[3] = is;
  int count_cons_mem[4];
  count_cons_mem[0] = 1;
  count_cons_mem[1] = block_size.nx3;
  count_cons_mem[2] = block_size.nx2;
  count_cons_mem[3] = block_size.nx1;

// reading attributes:
// H5File file( input_filename.c_str(), H5F_ACC_RDONLY );
    /*
    hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
    hid_t file = H5Fopen(input_filename.c_str(), H5F_ACC_RDONLY, property_list_file);
    H5Pclose(property_list_file);
    // time:
    Real time;
    hid_t attr = H5Aopen(file, "Time", H5P_DEFAULT);
    H5Aread(attr, H5T_REAL, &time);
    H5Aclose(attr);
    std::cout << "time = " << time << "\n";
    getchar();
    */
    
    // reading the XYZ mesh parameters:
    Real coord_range1[3], coord_range2[3], coord_range3[3];
    int coord_ncells[3];
    
    HDF5TripleRealAttribute(input_filename.c_str(), "RootGridX1", coord_range1);
    HDF5TripleRealAttribute(input_filename.c_str(), "RootGridX2", coord_range2);
    HDF5TripleRealAttribute(input_filename.c_str(), "RootGridX3", coord_range3);
    HDF5TripleIntAttribute(input_filename.c_str(), "RootGridSize", coord_ncells);
    
    int numblocks = HDF5IntAttribute(input_filename.c_str(), "NumMeshBlocks");
// pin->time = Time; // updating time (should it work?)
    
   // AthenaArray<Real> x1f;
    
   // x1f.NewAthenaArray(257);
    
// HDF5ReadRealArray(input_filename.c_str(), "x1f", 1, 0, 257, 1, 0, 257, x1f, true);
std::cout << "coord_ncells = " << coord_ncells[0] << ".." << coord_ncells[1] << ".." << coord_ncells[2] << "\n";
//getchar();
    
    std::cout << "bgdrho = " << bgdrho << "\n";
    AthenaArray<Real> u_old;
    u_old.NewAthenaArray(numblocks, 5, coord_ncells[2]+2*NGHOST, coord_ncells[1]+2*NGHOST, coord_ncells[0]+2*NGHOST);
    
  // Set conserved values from file
    for (int n = 0; n < NHYDRO; ++n) {
        start_cons_file[0] = start_cons_indices[n];
        start_cons_mem[0] = n;
        HDF5ReadRealArray(input_filename.c_str(), dataset_cons.c_str(), 5, start_cons_file,
                          count_cons_file, 4, start_cons_mem,
                          count_cons_mem, u_old, true); // this sets the variables on the old mesh
        if(n == index_etot) std::cout << "E test = " << u_old(0,IEN, 100,100,100) << " or " << phydro->u(0,IEN, 100,100,100) << "\n";
        // std::cout << "ks = " << ks << "\n";
        // std::cout << "ke = " << ke << "\n";
        // getchar();
    }
        // new mesh:
       //  for (int kb = 0; kb < numblocks; ++kb){
//    for (int n = 0; n < NHYDRO; ++n) {
            for (int k = ks; k <= ke; k++) {
                for (int j = js; j <= je; j++) {
                    for (int i = is; i <= ie; i++) {
                        Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k);
                        Real x1 = x * std::cos(rotangle) + y * std::sin(rotangle), y1 = y * std::cos(rotangle) - x * std::sin(rotangle);
                        int kold = (z-coord_range3[0])/(coord_range3[1]-coord_range3[0]) * (double)coord_ncells[2];
                        int jold = (y1-coord_range2[0])/(coord_range2[1]-coord_range2[0]) * (double)coord_ncells[1];
                        int iold = (x1-coord_range1[0])/(coord_range1[1]-coord_range1[0]) * (double)coord_ncells[0];
                        if ((kold >= NGHOST ) && (kold < coord_ncells[2]-NGHOST) && (jold >= NGHOST ) && (jold < coord_ncells[1]-NGHOST) && (iold >= NGHOST ) && (iold < coord_ncells[0]-NGHOST)){
                            phydro->u(0,IDN, k, j, i) = u_old(0,IDN, kold, jold, iold);
                            if (NON_BAROTROPIC_EOS) {
                                phydro->u(0,IEN, k, j, i) = u_old(0,IEN, kold, jold, iold);
                                // 0.5 * (SQR(u_old(0,IM1, kold, jold, iold))+SQR(u_old(0,IM2, kold, jold, iold))+SQR(u_old(0,IM3, kold, jold, iold))) / u_old(0,IDN, kold, jold, iold) + bgdp;
                                //nstd::cout << "thermal energy = " << u_old(0,IEN, kold, jold, iold) - 0.5 * (SQR(u_old(0,IM1, kold, jold, iold))+SQR(u_old(0,IM2, kold, jold, iold))+SQR(u_old(0,IM3, kold, jold, iold))) / u_old(0,IDN, kold, jold, iold) << "\n";
                            }
                            phydro->u(0,IM1, k, j, i) = u_old(0,IM1, kold, jold, iold);
                            phydro->u(0,IM2, k, j, i) = u_old(0,IM2, kold, jold, iold);
                            phydro->u(0,IM3, k, j, i) = u_old(0,IM3, kold, jold, iold);
                        }else{
                            phydro->u(0,IDN, k, j, i) = bgdrho;
                            phydro->u(0,IM1, k, j, i) = 0.;
                            phydro->u(0,IM2, k, j, i) = 0.;
                            phydro->u(0,IM3, k, j, i) = 0.;
                            if (NON_BAROTROPIC_EOS) {
                                phydro->u(0,4, k, j, i) = bgdp;
                            }
                        }
                    }
                }
            }
 //   }

    
    
  // Set field array selections
  int start_field_file[4];
  start_field_file[0] = gid;
  start_field_file[1] = 0;
  start_field_file[2] = 0;
  start_field_file[3] = 0;
  int count_field_file[4];
  count_field_file[0] = 1;
  int start_field_mem[3];
  start_field_mem[0] = ks;
  start_field_mem[1] = js;
  start_field_mem[2] = is;
  int count_field_mem[3];

  // Set magnetic field values from file
  if (MAGNETIC_FIELDS_ENABLED) {
    // Set B1
    count_field_file[1] = block_size.nx3;
    count_field_file[2] = block_size.nx2;
    count_field_file[3] = block_size.nx1 + 1;
    count_field_mem[0] = block_size.nx3;
    count_field_mem[1] = block_size.nx2;
    count_field_mem[2] = block_size.nx1 + 1;
    HDF5ReadRealArray(input_filename.c_str(), dataset_b1.c_str(), 4, start_field_file,
                      count_field_file, 3, start_field_mem,
                      count_field_mem, pfield->b.x1f, true);

    // Set B2
    count_field_file[1] = block_size.nx3;
    count_field_file[2] = block_size.nx2 + 1;
    count_field_file[3] = block_size.nx1;
    count_field_mem[0] = block_size.nx3;
    count_field_mem[1] = block_size.nx2 + 1;
    count_field_mem[2] = block_size.nx1;
    HDF5ReadRealArray(input_filename.c_str(), dataset_b2.c_str(), 4, start_field_file,
                      count_field_file, 3, start_field_mem,
                      count_field_mem, pfield->b.x2f, true);

    // Set B3
    count_field_file[1] = block_size.nx3 + 1;
    count_field_file[2] = block_size.nx2;
    count_field_file[3] = block_size.nx1;
    count_field_mem[0] = block_size.nx3 + 1;
    count_field_mem[1] = block_size.nx2;
    count_field_mem[2] = block_size.nx1;
    HDF5ReadRealArray(input_filename.c_str(), dataset_b3.c_str(), 4, start_field_file,
                      count_field_file, 3, start_field_mem,
                      count_field_mem, pfield->b.x3f, true);
  }

  // Make no-op collective reads if using MPI and ranks have unequal numbers of blocks
#ifdef MPI_PARALLEL
  {
    int num_blocks_this_rank = pmy_mesh->nblist[Globals::my_rank];
    if (lid == num_blocks_this_rank - 1) {
      int block_shortage_this_rank = 0;
      for (int rank = 0; rank < Globals::nranks; ++rank) {
        block_shortage_this_rank =
            std::max(block_shortage_this_rank,
                     pmy_mesh->nblist[rank] - num_blocks_this_rank);
      }
      for (int block = 0; block < block_shortage_this_rank; ++block) {
        for (int n = 0; n < NHYDRO; ++n) {
          start_cons_file[0] = start_cons_indices[n];
          start_cons_mem[0] = n;
          HDF5ReadRealArray(input_filename.c_str(), dataset_cons.c_str(), 5,
                            start_cons_file, count_cons_file, 4,
                            start_cons_mem, count_cons_mem,
                            phydro->u, true, true);
        }
        if (MAGNETIC_FIELDS_ENABLED) {
          count_field_file[1] = block_size.nx3;
          count_field_file[2] = block_size.nx2;
          count_field_file[3] = block_size.nx1 + 1;
          count_field_mem[0] = block_size.nx3;
          count_field_mem[1] = block_size.nx2;
          count_field_mem[2] = block_size.nx1 + 1;
          HDF5ReadRealArray(input_filename.c_str(), dataset_b1.c_str(), 4,
                            start_field_file, count_field_file, 3,
                            start_field_mem, count_field_mem,
                            pfield->b.x1f, true, true);
          count_field_file[1] = block_size.nx3;
          count_field_file[2] = block_size.nx2 + 1;
          count_field_file[3] = block_size.nx1;
          count_field_mem[0] = block_size.nx3;
          count_field_mem[1] = block_size.nx2 + 1;
          count_field_mem[2] = block_size.nx1;
          HDF5ReadRealArray(input_filename.c_str(), dataset_b2.c_str(), 4,
                            start_field_file, count_field_file, 3,
                            start_field_mem, count_field_mem,
                            pfield->b.x2f, true, true);
          count_field_file[1] = block_size.nx3 + 1;
          count_field_file[2] = block_size.nx2;
          count_field_file[3] = block_size.nx1;
          count_field_mem[0] = block_size.nx3 + 1;
          count_field_mem[1] = block_size.nx2;
          count_field_mem[2] = block_size.nx1;
          HDF5ReadRealArray(input_filename.c_str(), dataset_b3.c_str(), 4,
                            start_field_file, count_field_file, 3,
                            start_field_mem, count_field_mem,
                            pfield->b.x3f, true, true);
        }
      }
    }
  }
#endif
  return;
}
