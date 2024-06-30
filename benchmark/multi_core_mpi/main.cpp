#include "strat.hpp"
#include "mpi.h"
#include "utils.hpp"
#include <chrono>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int size;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double* distr_A;
    double* rhs;
    double* x;

    int* rows_per_process;
    int* displacements;

    size_t rows;
    size_t cols;
    size_t v_cols;

    if (utils::mpi::mpi_distributed_read_matrix(argv[1], distr_A, rows, cols, rows_per_process, displacements))
    {
        if (rank == 0)
            std::cout << "Everything done correctly for matrix" << std::endl;
    }
    if (utils::mpi::mpi_distributed_read_all_vector(argv[2], rhs, rows, v_cols, rows_per_process, displacements))
    {
        if (rank == 0)
            std::cout << "Everything done correctly for vector" << std::endl;
    }

    x = new double[cols];
    auto t1 = std::chrono::high_resolution_clock::now();
    if (rank == 0)
        std::cout << "Calling conjugate gradient" << std::endl;
    conjugate_gradient(distr_A, rhs, x, rows, cols, rows_per_process, displacements, 100000, 1.0e-6);
    auto t2 = std::chrono::high_resolution_clock::now();

    if (rank == 0)
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;

    delete [] distr_A;
    delete [] rhs;
    delete [] x;

    delete [] rows_per_process;
    delete [] displacements;

    MPI_Finalize();
}