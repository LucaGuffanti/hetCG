#ifndef CUDA_MPI_HPP
#define CUDA_MPI_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <unistd.h>
#include <stdint.h>
#include <mpi.h>

// Gatekeeping functions for cuda device identification.
static uint64_t getHostHash(const char* string);
static void getHostName(char* hostname, int maxlen);

// This method assumes that every mpi rank has loaded correctly data using the provided methods
void conjugate_gradient(const double* distr_A, double* x, double* b, const unsigned int rows, const unsigned int cols, const int* rows_per_process, const int* displacements, const int max_iter, const double rel_err); 
#endif