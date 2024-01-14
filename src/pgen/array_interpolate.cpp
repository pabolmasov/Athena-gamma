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
#ifdef FFT
#include "../fft/athena_fft.hpp"     // Fourier transforms
#endif
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
//!
#ifdef FFT
fftw_plan fplan_forward;
fftw_plan fplan_backward;
fftw_complex *Fphi;
#endif

int RefinementCondition(MeshBlock *pmb);


namespace{
    Real bgdrho = 1e-8, bgdp = 1e-8, bgdvx, bgdvy, bgdvz, rotangle = 0., addvx, addvy, addvz;
    Real int_linear(AthenaArray<Real>& u, int index, Real kest, Real jest, Real iest);
    Real int_linear(AthenaArray<Real>& u, int index1, int index2, Real kest, Real jest, Real iest); // one index more

    Real x1min;
    Real threshold;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
    // x1min = pin->GetReal("mesh","x1min");
    std::cerr << "xmin = " << x1min << "\n";
    // getchar();
    bgdrho = pin->GetReal("problem","bgdrho");
    bgdp = pin->GetReal("problem","bgdp");
    bgdvx = pin->GetOrAddReal("problem","bgdvx", 0.0);
    bgdvy = pin->GetOrAddReal("problem","bgdvy", 0.0);
    bgdvz = pin->GetOrAddReal("problem","bgdvz", 0.0);
    addvx = pin->GetOrAddReal("problem","addvx", 0.0);
    addvy = pin->GetOrAddReal("problem","addvy", 0.0);
    addvz = pin->GetOrAddReal("problem","addvz", 0.0);
    rotangle = pin->GetOrAddReal("problem","rotangle", 0.);
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("problem","four_pi_G");
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);
  }
    if (adaptive) {
      EnrollUserRefinementCondition(RefinementCondition);
      threshold = pin->GetReal("problem", "thr");
    }

}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    
    Real omega_Jacobi = pin->GetOrAddReal("problem","omega_Jacobi", 1.);
    Real div_tol = pin->GetOrAddReal("problem","div_tol", 1.e-3);
    Real gamma = pin->GetReal("hydro","gamma");
    Real dfloor = pin->GetReal("hydro","dfloor");
    // Determine locations of initial values
    std::string input_filename = pin->GetString("problem", "input_filename");
    std::string b_input_filename = pin->GetString("problem", "B_input_filename");
    std::string dataset_cons = pin->GetString("problem", "dataset_cons");
    int index_dens = pin->GetInteger("problem", "index_dens");
    int index_mom1 = pin->GetInteger("problem", "index_mom1");
    int index_mom2 = pin->GetInteger("problem", "index_mom2");
    int index_mom3 = pin->GetInteger("problem", "index_mom3");
    int index_etot = pin->GetInteger("problem", "index_etot");
    int index_b1 = pin->GetInteger("problem", "index_b1");
    int index_b2 = pin->GetInteger("problem", "index_b2");
    int index_b3 = pin->GetInteger("problem", "index_b3");
    std::string dataset_b1 = pin->GetString("problem", "dataset_b1");
    std::string dataset_b2 = pin->GetString("problem", "dataset_b2");
    std::string dataset_b3 = pin->GetString("problem", "dataset_b3");
    
    Real coord_range1[3], coord_range2[3], coord_range3[3];
    int coord_ncells[3], coord_blockcells[3];
    
    HDF5TripleRealAttribute(input_filename.c_str(), "RootGridX1", coord_range1);
    HDF5TripleRealAttribute(input_filename.c_str(), "RootGridX2", coord_range2);
    HDF5TripleRealAttribute(input_filename.c_str(), "RootGridX3", coord_range3);
    // HDF5TripleIntAttribute(input_filename.c_str(), "RootGridSize", coord_ncells);
    HDF5TripleIntAttribute(input_filename.c_str(), "MeshBlockSize", coord_blockcells);
    int numblocks = HDF5IntAttribute(input_filename.c_str(), "NumMeshBlocks");

    // Set conserved array selections
    int start_cons_file[5];
    // [0] is the id of variable
    start_cons_file[1] = 0; // gid is core/thread ID? this is current core ID for the new run
    start_cons_file[2] = 0;
    start_cons_file[3] = 0;
    start_cons_file[4] = 0;
    // std::cout << "start cons = " << start_cons_file[0] << " " << start_cons_file[1] << " " << start_cons_file[2] << " " << start_cons_file[3]  << " " << start_cons_file[4] << "\n";
    // getchar();
    int start_cons_indices[5];
    start_cons_indices[IDN] = index_dens;
    start_cons_indices[IM1] = index_mom1;
    start_cons_indices[IM2] = index_mom2;
    start_cons_indices[IM3] = index_mom3;
    start_cons_indices[IEN] = index_etot;

    int count_cons_file[5];
    count_cons_file[0] = 1;
    count_cons_file[1] = numblocks;
    count_cons_file[2] = coord_blockcells[2]; // block_size.nx3;
    count_cons_file[3] = coord_blockcells[1]; // block_size.nx2;
    count_cons_file[4] = coord_blockcells[0]; // block_size.nx1;
    
    std::cout << "to " <<  block_size.nx1 << " from " << coord_blockcells[2] << "\n";
    std::cout << "to " << block_size.nx2 << " from " << coord_blockcells[1] << "\n";
    std::cout << "to " << block_size.nx3 << " from " << coord_blockcells[0] << "\n";
    // getchar();
    
    int start_cons_mem[5];
    // [0] is the id
    // [1] is block
    start_cons_mem[0] = 0;
    start_cons_mem[1] = 0;
    start_cons_mem[2] = ks; // should it be -NGHOST or something?
    start_cons_mem[3] = js;
    start_cons_mem[4] = is;
    int count_cons_mem[5];
    count_cons_mem[0] = 1;
    count_cons_mem[1] = numblocks;
    count_cons_mem[2] = coord_blockcells[2]; // block_size.nx3;
    count_cons_mem[3] = coord_blockcells[1]; // block_size.nx2;
    count_cons_mem[4] = coord_blockcells[0]; // block_size.nx1;
    
    std::cout << "core " << gid << "\n";
    
    std::cout << "coord_ncells = " << coord_blockcells[0] << ".." << coord_blockcells[1] << ".." << coord_blockcells[2] << "\n";
    
    std::cout << "bgdrho = " << bgdrho << "\n";
    std::cout << "numblocks = " << numblocks << "\n";
    
    // parameters of the coordinate arrays:
    int start_x_file[2], start_y_file[2], start_z_file[2];
    start_x_file[1] = 0; start_x_file[0] = 0;
    start_y_file[1] = 0; start_y_file[0] = 0;
    start_z_file[1] = 0; start_z_file[0] = 0;
    int count_x_file[2], count_y_file[2], count_z_file[2];
    count_x_file[0] = numblocks; count_x_file[1] = coord_blockcells[2];
    count_y_file[0] = numblocks; count_y_file[1] = coord_blockcells[1];
    count_z_file[0] = numblocks; count_z_file[1] = coord_blockcells[0];
    int start_x_mem[2], start_y_mem[2], start_z_mem[2];
    start_x_mem[0] = 0;  start_x_mem[1] = 0;
    start_y_mem[0] = 0;  start_y_mem[1] = 0;
    start_z_mem[0] = 0;  start_z_mem[1] = 0;
    int count_x_mem[2], count_y_mem[2], count_z_mem[2];
    count_x_mem[0] = numblocks; count_x_mem[1] = coord_blockcells[2];
    count_y_mem[0] = numblocks; count_y_mem[1] = coord_blockcells[1];
    count_z_mem[0] = numblocks; count_z_mem[1] = coord_blockcells[0];
    
    // coordinate arrays
    AthenaArray<Real> x1v_old;
    x1v_old.NewAthenaArray(numblocks, coord_blockcells[2]);
    AthenaArray<Real> x2v_old;
    x2v_old.NewAthenaArray(numblocks, coord_blockcells[1]);
    AthenaArray<Real> x3v_old;
    x3v_old.NewAthenaArray(numblocks, coord_blockcells[0]);
    HDF5ReadRealArray(input_filename.c_str(), "x1v", 2, start_x_file,
                      count_x_file, 2, start_x_mem,
                      count_x_mem, x1v_old, true); // this sets the old mesh
    HDF5ReadRealArray(input_filename.c_str(), "x2v", 2, start_y_file,
                      count_y_file, 2, start_y_mem,
                      count_y_mem, x2v_old, true); // this sets the old mesh
    HDF5ReadRealArray(input_filename.c_str(), "x3v", 2, start_z_file,
                      count_z_file, 2, start_z_mem,
                      count_z_mem, x3v_old, true); // this sets the old mesh
    
    // coordinate arrays: faces
    AthenaArray<Real> x1f_old;
    x1f_old.NewAthenaArray(numblocks, coord_blockcells[2]+1);
    AthenaArray<Real> x2f_old;
    x2f_old.NewAthenaArray(numblocks, coord_blockcells[1]+1);
    AthenaArray<Real> x3f_old;
    x3f_old.NewAthenaArray(numblocks, coord_blockcells[0]+1);
    count_x_file[1] = coord_blockcells[2]+1;
    count_x_mem[1] = coord_blockcells[2]+1;
    count_y_file[1] = coord_blockcells[1]+1;
    count_y_mem[1] = coord_blockcells[1]+1;
    count_z_file[1] = coord_blockcells[0]+1;
    count_z_mem[1] = coord_blockcells[0]+1;
    std::cout << "reading face arrays \n";
    HDF5ReadRealArray(input_filename.c_str(), "x1f", 2, start_x_file,
                      count_x_file, 2, start_x_mem,
                      count_x_mem, x1f_old, true); // this sets the old mesh
    HDF5ReadRealArray(input_filename.c_str(), "x2f", 2, start_y_file,
                      count_y_file, 2, start_y_mem,
                      count_y_mem, x2f_old, true); // this sets the old mesh
    HDF5ReadRealArray(input_filename.c_str(), "x3f", 2, start_z_file,
                      count_z_file, 2, start_z_mem,
                      count_z_mem, x3f_old, true); // this sets the old mesh
    // getchar();
    AthenaArray<Real> u_old;
    u_old.NewAthenaArray(5, numblocks, coord_blockcells[2]+2*NGHOST, coord_blockcells[1]+2*NGHOST, coord_blockcells[0]+2*NGHOST);
    
    AthenaArray<int> npoints;
    npoints.NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);

    int nvars = NHYDRO;
    //    if(MAGNETIC_FIELDS_ENABLED) nvars += 3;
    
    //    for (int kb = 0; kb < numblocks; ++kb){
    //      start_cons_file[1] = kb;
        for (int n = 0; n < nvars; ++n) {
            start_cons_file[0] = start_cons_indices[n];
            start_cons_mem[0] = n;
            HDF5ReadRealArray(input_filename.c_str(), dataset_cons.c_str(), 5, start_cons_file,
                              count_cons_file, 5, start_cons_mem, count_cons_mem, u_old, true); // this sets the variables on the old mesh
         }
    
    //getchar();
    
    AthenaArray<Real> MF_old; // 1, MF_old2, MF_old3; // magnetic fields (face-centered)
    int start_MF_file[5];
    int count_MF_file[5];
    int start_MF_mem[5];
    int count_MF_mem[5];

// MF divergence control:
    // AthenaArray<Real> divB; // 1, MF_old2, MF_old3; // magnetic fields (face-centered)
    // Real divB[block_size.nx1][block_size.nx2][block_size.nx3];
    // phi = divB; // pointing to divB
    
    if(MAGNETIC_FIELDS_ENABLED){
        MF_old.NewAthenaArray(3, numblocks, coord_blockcells[2]+2*NGHOST+1, coord_blockcells[1]+2*NGHOST+1, coord_blockcells[0]+2*NGHOST+1);
        // divB.NewAthenaArray(block_size.nx1, block_size.nx2, block_size.nx3);
        // MF_old2.NewAthenaArray(3, numblocks, coord_blockcells[2]+2*NGHOST, coord_blockcells[1]+2*NGHOST+1, coord_blockcells[0]+2*NGHOST);
        // MF_old3.NewAthenaArray(3, numblocks, coord_blockcells[2]+2*NGHOST, coord_blockcells[1]+2*NGHOST, coord_blockcells[0]+2*NGHOST+1);

        start_MF_file[0] = 0;  start_MF_mem[0] = 0;
        start_MF_file[1] = 0;  start_MF_mem[1] = 0;
        
        count_MF_file[0] = 1;  count_MF_mem[0] = 1;
        count_MF_file[1] = numblocks;  count_MF_mem[1] = numblocks;
        
        start_MF_file[2] = 0;
        start_MF_file[3] = 0;
        start_MF_file[4] = 0;

        start_MF_mem[2] = ks;
        start_MF_mem[3] = js;
        start_MF_mem[4] = is;

        // count_MF_file[2] = coord_blockcells[2]+1; // block_size.nx3;
        // count_MF_file[3] = coord_blockcells[1]+1; // block_size.nx2;
        // count_MF_file[4] = coord_blockcells[0]+1; // block_size.nx1;
        
        //count_MF_mem[2] = coord_blockcells[2]+1; // block_size.nx3;
        //count_MF_mem[3] = coord_blockcells[1]+1; // block_size.nx2;
        //count_MF_mem[4] = coord_blockcells[0]+1; // block_size.nx1;

        std::cout << "reading MF "<< dataset_b1.c_str()<<  " \n";
        count_MF_file[2] = coord_blockcells[2];
        count_MF_file[3] = coord_blockcells[1];
        count_MF_file[4] = coord_blockcells[0];
        count_MF_mem[2] = coord_blockcells[2];
        count_MF_mem[3] = coord_blockcells[1];
        count_MF_mem[4] = coord_blockcells[0];
        HDF5ReadRealArray(b_input_filename.c_str(), dataset_b1.c_str(), 5, start_MF_file,
                          count_MF_file, 5, start_MF_mem, count_MF_mem, MF_old, true); // this sets the variables on the old mesh
        start_MF_file[0] = 1;  start_MF_mem[0] =1;
        std::cout << "reading MF "<< dataset_b2.c_str() << " \n";
        count_MF_file[2] = coord_blockcells[2];
        count_MF_file[3] = coord_blockcells[1] ;
        count_MF_file[4] = coord_blockcells[0];
        count_MF_mem[2] = coord_blockcells[2];
        count_MF_mem[3] = coord_blockcells[1] ;
        count_MF_mem[4] = coord_blockcells[0];
        HDF5ReadRealArray(b_input_filename.c_str(), dataset_b2.c_str(), 5, start_MF_file,
                          count_MF_file, 5, start_MF_mem, count_MF_mem, MF_old, true); // this sets the variables on the old mesh
        start_MF_file[0] = 2; start_MF_mem[0] = 2;
        std::cout << "reading MF "<< dataset_b3.c_str()<< " \n";
        count_MF_file[2] = coord_blockcells[2];
        count_MF_file[3] = coord_blockcells[1];
        count_MF_file[4] = coord_blockcells[0];
        count_MF_mem[2] = coord_blockcells[2];
        count_MF_mem[3] = coord_blockcells[1];
        count_MF_mem[4] = coord_blockcells[0];
        HDF5ReadRealArray(b_input_filename.c_str(), dataset_b3.c_str(), 5, start_MF_file,
                          count_MF_file, 5, start_MF_mem, count_MF_mem, MF_old, true); // this sets the variables on the old mesh
        //getchar();
    }
    //  getchar();

  // }
    // Set conserved values from file
    // new mesh:
        // int gid_old = kb;
        // tart_cons_file[1] = gid_old;
        // start_cons_mem[1] = gid_old;
        //    for (int n = 0; n < NHYDRO; ++n) {
        // each block has a uniform grid in X, Y, and Z
    
    for (int kb = 0; kb < numblocks; ++kb){
        Real xold_min = x1v_old(kb, 0), xold_max = x1v_old(kb, coord_blockcells[2]-1);
        Real yold_min = x2v_old(kb, 0), yold_max = x2v_old(kb, coord_blockcells[1]-1);
        Real zold_min = x3v_old(kb, 0), zold_max = x3v_old(kb, coord_blockcells[0]-1);
        Real xold_fmin = x1f_old(kb, 0), xold_fmax = x1f_old(kb, coord_blockcells[2]);
        Real yold_fmin = x2f_old(kb, 0), yold_fmax = x2f_old(kb, coord_blockcells[1]);
        Real zold_fmin = x3f_old(kb, 0), zold_fmax = x3f_old(kb, coord_blockcells[0]);
        Real dx_old = (xold_max-xold_min) / ((double)coord_blockcells[2]);
        Real dy_old = (yold_max-yold_min) / ((double)coord_blockcells[1]);
        Real dz_old = (zold_max-zold_min) / ((double)coord_blockcells[0]);
        std::cout << "block No " << kb << ":\n x = " << xold_min << ".." << xold_max << "\n";
        std::cout << "y = " << yold_min << ".." << yold_max << "\n" ;
        std::cout << "z = "  << zold_min << ".." << zold_max << "\n";
        std::cout << "Nx = " << coord_blockcells[2] << "; Ny = " << coord_blockcells[1] << "; Nz = "<< coord_blockcells[0]  << "\n";
        std::cout << "k = " << ks << ".." << ke << "\n";
        std::cout << "j = " << js << ".." << je << "\n";
        std::cout << "i = " << is << ".." << ie << "\n";

        // getchar();
        for (int k = ks-NGHOST; k < ke+NGHOST; k++) {
            for (int j = js-NGHOST; j < je+NGHOST; j++) {
                for (int i = is-NGHOST; i < ie+NGHOST; i++) {
                    Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k);
                    Real dx = pcoord->dx1v(i), dy = pcoord->dx2v(j), dz = pcoord->dx3v(k);
                    Real x1 = x * std::cos(rotangle) + y * std::sin(rotangle), y1 = y * std::cos(rotangle) - x * std::sin(rotangle);
                    
                    Real kold = (z-zold_min)/dz_old, jold = (y1-yold_min)/dy_old, iold = (x1-xold_min)/dx_old;
                    
                    // int kold = (int)std::round((z-zold_min)/dz_old) ; // /(zold_max-zold_min) * (double)(coord_blockcells[0]));
                    // int jold = (int)std::round((y1-yold_min)/dy_old) ; // (yold_max-yold_min) * (double)(coord_blockcells[1]));
                    // int iold = (int)std::round((x1-xold_min)/dx_old) ; // /(xold_max-xold_min) * ((double)coord_blockcells[2]));
                    kold = std::min(std::max(kold, 0.), (double)coord_blockcells[0]-1);
                    jold = std::min(std::max(jold, 0.), (double)coord_blockcells[1]-1);
                    iold = std::min(std::max(iold, 0.), (double)coord_blockcells[2]-1);
                    // if ((kold >= 0) && (kold < coord_blockcells[0]) && (jold >= 0) && (jold < coord_blockcells[1]) && (iold >= 0) && (iold < coord_blockcells[2]))
                    Real dd = int_linear(u_old, IDN, kb, kold, jold, iold);
                   //if((x1>(xold_fmin-dx)) && (x1 < (xold_fmax+dx)) && (y1>(yold_fmin-dy)) && (y1 < (yold_fmax+dy)) &&(z>(zold_fmin-dz)) && (z < (zold_fmax+dz))){
                    if(dd>dfloor){
                    // std::cout << "iold = " << iold << "; jold = " << jold << "; kold = " << kold << "\n" ;
                        
                        phydro->u(IDN, k, j, i) += dd;
                        // u_old(IDN, kb, kold, jold, iold);
                        if (NON_BAROTROPIC_EOS) {
                            phydro->u(IEN, k, j, i) += int_linear(u_old, IEN, kb, kold, jold, iold) + dd * (SQR(addvx) + SQR(addvy)+SQR(addvz))/2.;
                            // u_old(IEN, kb, kold, jold, iold)
                                                   // + u_old(IDN, kb, kold, jold, iold) * (SQR(addvx) + SQR(addvy)+SQR(addvz))/2.;
                        }
                        phydro->u(IM1, k, j, i) += int_linear(u_old, IM1, kb, kold, jold, iold) + dd * addvx;
                        // u_old(IM1, kb, kold, jold, iold) + u_old(IDN, kb, kold, jold, iold) * addvx;
                        phydro->u(IM2, k, j, i) += int_linear(u_old, IM2, kb, kold, jold, iold) + dd * addvy;
                        // u_old(IM2, kb, kold, jold, iold) + u_old(IDN, kb, kold, jold, iold) * addvy;
                        phydro->u(IM3, k, j, i) += int_linear(u_old, IM3, kb, kold, jold, iold) + dd * addvz;
                        // u_old(IM3, kb, kold, jold, iold) + u_old(IDN, kb, kold, jold, iold) * addvz;
                        npoints(k,j,i) ++;
                        if(MAGNETIC_FIELDS_ENABLED){
                            pfield->b.x1f(k,j,i) += int_linear(MF_old, 0, kb, kold, jold, iold);
                            // MF_old(kb, 0, kold, jold, iold);
                            pfield->b.x2f(k,j,i) += int_linear(MF_old, 1, kb, kold, jold, iold);
                            // MF_old(kb, 1, kold, jold, iold);
                            pfield->b.x3f(k,j,i) += int_linear(MF_old, 2, kb, kold, jold, iold);
                            // MF_old(kb, 2, kold, jold, iold);
                        }
                    }
                }
            }
            // std::cout << "k = " << k << "\n";
            // getchar();
        }
    }
    // std::cout << "interpolated \n";
    int maxnpoints = 0;

    // normalize by the number of points:
    for (int k = ks-NGHOST; k <= ke+NGHOST; k++) {
        for (int j = js-NGHOST; j <= je+NGHOST; j++) {
            for (int i = is-NGHOST; i <= ie+NGHOST; i++) {
                if (npoints(k,j,i) > 0){
                    maxnpoints = std::max(npoints(k,j,i), maxnpoints);
                    phydro->u(IDN, k, j, i) /= (double)npoints(k,j,i) ;
                    phydro->u(IM1, k, j, i) /= (double)npoints(k,j,i) ;
                    phydro->u(IM2, k, j, i) /= (double)npoints(k,j,i) ;
                    phydro->u(IM3, k, j, i) /= (double)npoints(k,j,i) ;
                    if (NON_BAROTROPIC_EOS) {
                        phydro->u(IEN, k, j, i) /= (double)npoints(k,j,i) ;
                    }
                    if(MAGNETIC_FIELDS_ENABLED){
                        pfield->b.x1f(k,j,i) /= (double)npoints(k,j,i) ;
                        pfield->b.x2f(k,j,i) /= (double)npoints(k,j,i) ;
                        pfield->b.x3f(k,j,i) /= (double)npoints(k,j,i) ;
                    }
                    }
                else{
                    phydro->u(IDN, k, j, i) = bgdrho;
                    phydro->u(IM1, k, j, i) = bgdrho * bgdvx;
                    phydro->u(IM2, k, j, i) = bgdrho * bgdvy;
                    phydro->u(IM3, k, j, i) = bgdrho * bgdvz;
                    if (NON_BAROTROPIC_EOS) {
                        phydro->u(IEN, k, j, i) = bgdp / (gamma-1.) + bgdrho * (SQR(bgdvx)+SQR(bgdvy)+SQR(bgdvz))/2.;
                    }
                }
            }
        }
    }
    
    // std::cout << "normalized \n";
    // pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST);

    // calculate divB (rectangular grid!)
    // double *in1 = (double *)fftw_malloc(sizeof(double) * N*N);
    AthenaArray<Real> divB;  // , phi, phi1;
    divB.NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
    // phi.NewAthenaArray(block_size.nx1, block_size.nx2, block_size.nx3);
    // phi1.NewAthenaArray(block_size.nx1, block_size.nx2, block_size.nx3);
    // double *divB = (double *)fftw_malloc(sizeof(double) * block_size.nx1 * block_size.nx2 * block_size.nx3);
    Real dx = pcoord->dx1v(0), dy = pcoord->dx2v(0), dz = pcoord->dx3v(0);

    if(MAGNETIC_FIELDS_ENABLED){
        int divborder = NGHOST;
        for (int k = ks-divborder; k < ke+divborder; k++) {
            for (int j = js-divborder; j <= je+divborder; j++) {
                for (int i = is-divborder; i < ie+divborder; i++) {
                    // Real dx = pcoord->dx1v(i), dy = pcoord->dx2v(j), dz = pcoord->dx3v(k);
                    divB(k,j,i) = (pfield->b.x1f(k,j,i+1)-pfield->b.x1f(k,j,i))/dx
                    + (pfield->b.x2f(k,j+1,i)-pfield->b.x2f(k,j,i))/dy
                    + (pfield->b.x3f(k+1,j,i)-pfield->b.x3f(k,j,i))/dz;
                    // phi(k,j,i) = divB(k,j,i) * 1.0;
                }
            }
        }
        
        std::cout << "divergence cleaning\n";
        // now solving an iterative scheme for diffusing out the gradient part of B
        int ctr=0; // number of relaxation iterations
        Real tol = div_tol, omega = omega_Jacobi * std::min(SQR(dx),std::min(SQR(dy), SQR(dz)));
        
        std::cout << "omega for iterations = " << omega << "\n";
        
        Real Berror=1., Bmax=1.;

        //for (int q = 0; q < niter; q++){ // iterations
        while(Berror > (tol*Bmax)){
            Berror=0.; Bmax = 0.;
            // writing phi1
            for (int k = ks-divborder+1; k <= ke+divborder; k++) {
                for (int j = js-divborder+1; j <= je+divborder; j++) {
                    for (int i = is-divborder+1; i <= ie+divborder; i++) {
                        if((k<ke)&&(j<je))pfield->b.x1f(k,j,i) += (divB(k,j,i)-divB(k,j,i-1))/dx * omega;
                        if((k<ke)&&(i<ie))pfield->b.x2f(k,j,i) += (divB(k,j,i)-divB(k,j-1,i))/dy * omega;
                        if((i<ie)&&(j<je))pfield->b.x3f(k,j,i) += (divB(k,j,i)-divB(k-1,j,i))/dz * omega;
                        
                        if ((k > ks) && (k < ke) && (j > js) && (j < je) && (i > is) && (i < ie)){
                            Bmax = std::max(Bmax,std::abs(pfield->b.x2f(k,j,i)));
                            Berror = std::max(Berror,std::abs(divB(k,j,i)));
                        }
                   }
                }
            }
#ifdef MPI_PARALLEL
            MPI_Allreduce(MPI_IN_PLACE, &Bmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &Berr, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
            pfield->fbvar.SetBoundaries();
            // rewriting divB
            for (int k = ks-divborder; k < ke+divborder; k++) {
                for (int j = js-divborder; j <= je+divborder; j++) {
                    for (int i = is-divborder; i < ie+divborder; i++) {
                        // Real dx = pcoord->dx1v(i), dy = pcoord->dx2v(j), dz = pcoord->dx3v(k);
                        divB(k,j,i) = (pfield->b.x1f(k,j,i+1)-pfield->b.x1f(k,j,i))/dx
                        + (pfield->b.x2f(k,j+1,i)-pfield->b.x2f(k,j,i))/dy
                        + (pfield->b.x3f(k+1,j,i)-pfield->b.x3f(k,j,i))/dz;
                    }
                }
            }
            if (ctr%100==0)
                std::cout << "div B error / B max = " << Berror << "/ " << Bmax << " = " << Berror/Bmax <<   "\n";
            ctr ++;
            //getchar();
        }
    }
/*
#ifdef FFT
        // FFT divergence cleaning
        Fphi = new fftw_complex[block_size.nx1,block_size.nx2, block_size.nx3/2+1];
        //fplan =  fftw_plan_dft_1d(, fft_data, fft_data, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan fplan_forward = fftw_plan_dft_r2c_3d(block_size.nx1,block_size.nx2, block_size.nx3,
                                       divB, Fphi,FFTW_FORWARD);
        fftw_plan fplan_backward = fftw_plan_dft_c2r_3d(block_size.nx1,block_size.nx2, block_size.nx3,
                                       Fphi, divB,FFTW_BACKWARD);

        fftw_execute(fplan_forward); // converting divB to Fourier space
        
        Real xrange = pcoord->dx1v(0) * (double)block_size.nx3, yrange = pcoord->dx1v(1) * (double)block_size.nx2, zrange = pcoord->dx1v(2) * (double)block_size.nx1;
        
        int kmid = ceil (block_size.nx1/2);
        int jmid = ceil (block_size.nx2/2);
        int imid = ceil (block_size.nx3/2);
        
        for (int k = ks; k <= ke; k++) {
            for (int j = js; j <= je; j++) {
                for (int i = is; i <= ie; i++) {
                    int bindex = k + block_size.nx1 * (j +  block_size.nx2 * i);
                    Real ksq = ((double)SQR(k-kmid)/SQR(zrange)+SQR(j-jmid)/SQR(yrange)+SQR(i-imid)/SQR(xrange));
                    Fphi[bindex][0] /= (double)ksq * (double)(block_size.nx3 * block_size.nx2 * block_size.nx1); // now it should become phi
                }
            }
        }
        fftw_execute(fplan_backward); // converting divB to Fourier space

        for (int k = ks; k <= ke+1; k++) {
            for (int j = js; j <= je+1; j++) {
                for (int i = is; i <= ie+1; i++) {
                    int bindex = k + block_size.nx1 * (j +  block_size.nx2 * i),
                        bindex_i0 = k + block_size.nx1 * (j +  block_size.nx2 * (i-1)),
                        bindex_j0 = k + block_size.nx1 * (j-1 +  block_size.nx2 * i),
                        bindex_k0 = k-1 + block_size.nx1 * (j +  block_size.nx2 * i);
                    Real dx = pcoord->dx1v(i), dy = pcoord->dx2v(j), dz = pcoord->dx3v(k);
                    pfield->b.x1f(k,j,i) -= (divB[bindex]-divB[bindex_i0]) / dx ;
                    pfield->b.x2f(k,j,i) -= (divB[bindex]-divB[bindex_j0]) / dy ;
                    pfield->b.x3f(k,j,i) -= (divB[bindex]-divB[bindex_k0]) / dz ;
                }
            }
        }
        fftw_destroy_plan(fplan_forward);
        fftw_destroy_plan(fplan_backward);
#endif
    }
*/
    std::cout << "max N = " << maxnpoints << "\n";
    // getchar();
/*
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
 */
  return;
}

namespace{

Real int_linear(AthenaArray<Real> &u, int index, Real kest, Real jest, Real iest){
    // version for one foreindex
    int k0 = (int)floor(kest), j0 = (int)floor(jest), i0 = (int)floor(iest);
    int k1 = k0+1, j1 = j0+1, i1 = i0+1;
    
    Real cz = std::fmod(kest,k0), cy = std::fmod(jest,j0), cx = std::fmod(iest,i0);
    
    Real unew0 = u(index, k0,j0,i0) + cy * (u(index, k0,j1,i0) - u(index, k0,j0,i0)) +  cx * (u(index, k0,j0,i1) - u(index, k0,j0,i0))
    + cy * cx * (u(index, k0,j1,i1) - u(index, k0,j0,i1) - u(index, k0,j1,i0) + u(index, k0,j0,i0)); // 2D interpolation at z = 0
    Real unew1 = u(index, k1,j0,i0) + cy * (u(index, k1,j1,i0) - u(index, k1,j0,i0)) +  cx * (u(index, k1,j0,i1) - u(index, k1,j0,i0))
    + cy * cx * (u(index, k1,j1,i1) - u(index, k1,j0,i1) - u(index, k1,j1,i0) + u(index, k1,j0,i0)); // 2D interpolation at z = 0
    
    return unew0 + cz * (unew1 - unew0);
}

Real int_linear(AthenaArray<Real> &u, int index1, int index2, Real kest, Real jest, Real iest){
    // version for two foreindices
    int k0 = (int)floor(kest), j0 = (int)floor(jest), i0 = (int)floor(iest);
    int k1 = k0+1, j1 = j0+1, i1 = i0+1;
    Real cz = std::fmod(kest,k0), cy = std::fmod(jest,j0), cx = std::fmod(iest,i0);

    if (kest >= 130.){
        std::cout << "raw frac: " << kest << ", " << jest << ", " << iest << "\n";
        std::cout << "int: " << k0 << ", " << j0 << ", " << i0 << "\n";
        std::cout << "fracs: " << cz << ", " << cy << ", " << cx << "\n";
    }

    Real& u00 = u(index1, index2, k0,j0,i0);
    
    Real unew0 = u00 + cy * (u(index1, index2,  k0,j1,i0) - u00) +  cx * (u(index1, index2, k0,j0,i1) - u00)
    + cy * cx * (u(index1, index2, k0,j1,i1) - u(index1, index2, k0,j0,i1) - u(index1, index2, k0,j1,i0) + u00); // 2D interpolation at z = 0
    u00 = u(index1, index2, k1,j0,i0);
    Real unew1 = u00 + cy * (u(index1, index2, k1,j1,i0) - u00) +  cx * (u(index1, index2, k1,j0,i1) - u00)
    + cy * cx * (u(index1, index2, k1,j1,i1) - u(index1, index2, k1,j0,i1) - u(index1, index2, k1,j1,i0) + u00); // 2D interpolation at z = 0
    
    return unew0 + cz * (unew1 - unew0);
}
}

int RefinementCondition(MeshBlock *pmb) {
    // from "slotted_cylinder"
    Real maxeps = 0.;
    // no refinement in adsence of MF
    if(MAGNETIC_FIELDS_ENABLED){
        for (int k=pmb->ks-1; k<=pmb->ke+1; k++) {
            for (int j=pmb->js-1; j<=pmb->je+1; j++) {
                for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
                    Real eps = std::sqrt(SQR(pmb->pfield->bcc(IB1, k,j,i+1) - pmb->pfield->bcc(IB1, k,j,i-1))
                                         + SQR(pmb->pfield->bcc(IB2, k,j+1,i) - pmb->pfield->bcc(IB2, k,j-1,i)));
                    // /r(n,k,j,i); Do not normalize by scalar, since (unlike IDN and IPR) there
                    // are are no physical floors / r=0 might be allowed. Compare w/ blast.cpp.
                    maxeps = std::max(maxeps, eps);
                }
            }
        }
    }
    // std::cout << "maxeps = " << maxeps << "\n";
  if (maxeps > threshold) return 1;
  if (maxeps < 0.25*threshold) return -1;
  return 0;
}
