#include <iostream>
#include "CUBLAS_single.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <random>
#include <climits>
#include "utils.hpp"

int main(int argc, char** argv)
{
    double* A; 
    double* x;
    double* b;

    size_t rows;
    size_t cols;
    size_t len;


    utils::read_matrix_from_file(argv[1], A, rows, cols);
    utils::read_vector_from_file(argv[2], b, len);

    auto t5 = std::chrono::high_resolution_clock::now();
    conjugate_gradient_blas(A, x, b, cols, 1000000, 1e-6);
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count() << std::endl;

    delete [] A;
    delete [] x;
    delete [] b;

}