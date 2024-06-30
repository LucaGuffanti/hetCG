#include <mpi.h>
#include <iostream>
#include <sstream>
#include <omp.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    int matrixDimension;
    double* globalMatrix;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_rank[size];
    int displacements[size];




    if (argc == 2)
    {
        matrixDimension = atoi(argv[1]);
    }

    if (rank == 0)
    {
        globalMatrix = new double[matrixDimension * matrixDimension];
        std::cout << "Building an SPD matrix of dimension " << matrixDimension << "x" << matrixDimension << std::endl;
    }

    for (int i = 0; i < size; i++)
    {
        rows_per_rank[i] = matrixDimension / size;
    }

    for (int i = 0; i < size; i++)
    {
        if (i < matrixDimension % size)
        {
            rows_per_rank[i]++;
        }
    }

    displacements[0] = 0;
    for(int i = 1; i < size; ++i)
    {
        displacements[i] = displacements[i-1] + rows_per_rank[i-1];
    }

    if (rank == 0)
    {
        std::cout << "The following rank to rows mapping is " << std::endl;
        for (int i = 0; i < size; i++)
        {
            std::cout << "Rank " << i << " has " << rows_per_rank[i] << " rows starting from " << displacements[i] << std::endl;
        }
    }
    
    double matrixLocal[rows_per_rank[rank] * matrixDimension] = {0};
    double vectorLocal[rows_per_rank[rank]];

    for (int i = 0; i < rows_per_rank[rank]; i++)
    {
        vectorLocal[i] = 1.0;
    }


    for (int i = 0; i < rows_per_rank[rank]; i++)
    {
        int currentrow = displacements[rank] + i;

        if (currentrow != 0)
            matrixLocal[i * matrixDimension + currentrow - 1] = 1.0;
        if (currentrow != matrixDimension - 1){
            matrixLocal[i * matrixDimension + currentrow + 1] = 1.0;
        }
        matrixLocal[i * matrixDimension + currentrow] = 2.0;
    }

    // Compute the actual displacements in term of elements
    int elements[size];
    for (int i = 0; i < size; i++)
    {
        elements[i] = rows_per_rank[i] * matrixDimension;
    }

    int elements_displacements[size];

    elements_displacements[0] = 0;
    for (int i = 1; i < size; i++)
    {
        elements_displacements[i] = elements_displacements[i - 1] + elements[i - 1];
    }

    MPI_Gatherv(matrixLocal, rows_per_rank[rank] * matrixDimension, MPI_DOUBLE, globalMatrix, elements, elements_displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "The global matrix is " << std::endl;
        for (int i = 0; i < matrixDimension; i++)
        {
            for (int j = 0; j < matrixDimension; j++)
            {
                std::cout << globalMatrix[i * matrixDimension + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_File fh_matrix;
    std::string matrix_name = "matrix"+std::to_string(matrixDimension)+".bin";
    MPI_File_open(MPI_COMM_WORLD, matrix_name.data(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_matrix);

    if (rank == 0){
        // The first rank will need to write the dimension of the matrix
        MPI_File_write(fh_matrix, &matrixDimension, 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_write(fh_matrix, &matrixDimension, 1, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_File_write_at_all(fh_matrix, sizeof(int) + sizeof(int) + elements_displacements[rank]*sizeof(double), matrixLocal, rows_per_rank[rank]*matrixDimension, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh_matrix);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_File fh_vector;
    std::string vector_name = "vector"+std::to_string(matrixDimension)+".bin";
    MPI_File_open(MPI_COMM_WORLD, vector_name.data(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_vector);

    if (rank == 0){
        // The first rank will need to write the dimension of the matrix
        MPI_File_write(fh_vector, &matrixDimension, 1, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_File_write_at_all(fh_vector, sizeof(int) + displacements[rank] * sizeof(double), vectorLocal, rows_per_rank[rank], MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh_vector);

    MPI_Finalize();

    return 0;
}   