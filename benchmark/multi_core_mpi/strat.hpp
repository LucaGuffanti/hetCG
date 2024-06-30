#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include <cmath>
#include <stdio.h>
#include <mpi.h>
#include <memory.h>
#include <iostream>
#include <cstdlib>


void run(const double * , const double * , double * , size_t , int , double )  ;

double dot(const double * x, const double * y, size_t size) ;
void axpby(double alpha, const double * x, double beta, double * y, size_t size) ;
void gemv(double alpha, const double * A, const double * x, double beta, double *& y, size_t num_rows, size_t num_cols, int displ);
void conjugate_gradient(const double * A, const double * b, double * x, size_t rows, size_t cols, int* rows_per_process, int* displacements, int max_iters, double rel_error);

#endif