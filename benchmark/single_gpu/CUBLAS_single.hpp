#ifndef CUBLAS_SINGLE_HPP
#define CUBLAS_SINGLE_HPP

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void conjugate_gradient_blas(double* A, double* x, double* b, int size, int max_iters, double rel_error);
#endif