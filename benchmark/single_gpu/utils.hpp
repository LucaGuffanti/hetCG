#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cassert>

#include <mpi.h>

#define SIZEOF_HEADER_ELEM sizeof(size_t)
#define FILE_HEADER_SIZE 2 * sizeof(size_t)

namespace utils{
    bool read_matrix_from_file(const char *, double *&, size_t &, size_t &);
    bool read_vector_from_file(const char * , double *& , size_t &);
    void create_vector(double * &, size_t , double );
    void create_matrix(double * &, size_t, size_t, double );
    bool read_matrix_rows(const char *, double *&, size_t , size_t , size_t &);
    bool read_matrix_dims(const char * , size_t &, size_t &);
    void print_matrix(const double * , size_t , size_t , FILE * = stdout);

    namespace mpi
    {
        /**
         * @brief Using MPI parallel IO operations, distributes a matrix read from file and computes
         * the offsets in terms of rows to distribute the load among the processors.
         * 
         * @param filename file to be read
         * @param matrix vector containing the matrix LOCAL TO EACH THREAD
         * @param rows number of rows of the matrix
         * @param cols number of columns of the matrix
         * @param rows_per_process number of rows associated to each process
         * @param displacements first row (from row 0) each process is associated to
         * @return true If everything worked 
         * @return false If the file could not be opened
         * 
         * @note It is assumed that MPI was correctly initialized. Additionally, the matrix, displacements and rows
         * vectors are initialized in the method
         */
        bool mpi_distributed_read_matrix(const char* filename, double*& matrix, size_t& rows, size_t& cols, int*& rows_per_process, int*& displacements);
        
        /**
         * @brief Using MPI parallel IO operations, reads and entire vector from file and computes
         * the offsets in terms of rows to distribute the load among the processors.
         * 
         * @param filename file to be read
         * @param vector vector containing the rhs
         * @param rows number of elements of the vector
         * @param cols number of columns. Will be 1.
         * @param rows_per_process number of rows associated to each process
         * @param displacements first row (from row 0) each process is associated to 
         * @return true If everything worked 
         * @return false If the file could not be opened
         * 
         * @note It is assumed that MPI was correctly initialized. Additionally, the vector, displacements and rows
         * vectors are initialized in the method.
         */
        bool mpi_distributed_read_all_vector(const char* filename, double*& vector, size_t& rows, size_t& cols, int*& rows_per_process, int*& displacements);

        /**
         * @brief Prints in parallel a global vector, subdividing it in chunks of rows. 
         * 
         * @param vector vector to be printed
         * @param rows number of elements of the vector 
         * @param rows_per_process number of elements associated to each process
         * @param displacements first element (with respect to element 0) associated to a process
         * 
         */
        void mpi_print_vector(const char* filename, const double* vector, const size_t rows, int* rows_per_process, int* displacements);
    }
}

#endif