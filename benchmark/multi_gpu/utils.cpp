#include "utils.hpp"

bool utils::read_matrix_from_file(const char * filename, double * &matrix_out, size_t &num_rows_out, size_t &num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }
    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    matrix_out = matrix;
    num_rows_out = num_rows;
    num_cols_out = num_cols;

    fclose(file);

    return true;
}

bool utils::read_vector_from_file(const char * filename, double * &vector_out, size_t &length)
{

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&length, sizeof(size_t), 1, file);
    vector_out = new double[length];
    fread(vector_out, sizeof(double), length, file);

    fclose(file);

    return true;
}

void utils::create_vector(double * &vector_out, size_t length, double scalar)
{

    vector_out = new double[length];

    for(size_t i = 0; i< length; i++)
        vector_out[i] = scalar;
}

void utils::create_matrix(double * &matrix_out, size_t n, size_t m, double scalar)
{

    matrix_out = new double[n*m];

    for(size_t r = 0; r<n; r++)
        for(size_t c = 0; c<m; c++)
            matrix_out[r*m + c] = scalar;
}

bool utils::read_matrix_rows(const char * filename, double * &matrix_out, size_t starting_row_num, size_t num_rows_to_read, size_t &num_cols)
{
    size_t num_rows;
    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "read_matrix_rows: Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    
    assert(starting_row_num + num_rows_to_read <= num_rows);

    matrix_out = new double[num_rows_to_read * num_cols];

    
    size_t offset = starting_row_num * num_cols + 2; 
    if (fseek(file, sizeof(double)*offset, SEEK_SET) != 0) {
        fprintf(stderr, "read_matrix_rows: Error setting file position");
        return false;
    }

    fread(matrix_out, sizeof(double), num_rows_to_read * num_cols, file);


    fclose(file);

    return true;
}



bool utils::read_matrix_dims(const char * filename, size_t &num_rows_out, size_t &num_cols_out)
{

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "read_matrix_dims: Cannot open output file\n");
        return false;
    }

    fread(&num_rows_out, sizeof(size_t), 1, file);
    fread(&num_cols_out, sizeof(size_t), 1, file);


    fclose(file);

    return true;
}

void utils::print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file )
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}

bool utils::mpi::mpi_distributed_read_matrix(const char* filename, double*& matrix, size_t& rows, size_t& cols, int*& rows_per_process, int*& displacements)
{
    int init_flag;
    MPI_Initialized(&init_flag );
    assert(init_flag && "ERROR: MPI WAS NOT INITIALIZED.");

    int rank;
    int size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_File file;
    MPI_Offset offset;
    MPI_Status status;

    size_t header[2];

    if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS)
    {
        return false;
    }

    // First each process reads the header of the file to get the 
    MPI_File_read_at_all(file, 0, header, 2, MPI_UNSIGNED_LONG_LONG, &status);
    rows = header[0];
    cols = header[1];

    // Then the work is spread among all ranks
    rows_per_process = new int[size];
    displacements = new int[size];

    for (unsigned int i = 0; i < size; ++i)
    {
        rows_per_process[i] = rows / size;
        if (i < rows % size)
        {
            rows_per_process[i]++;
        }
    }

    displacements[0] = 0;
    for (unsigned int i = 1; i < size; ++i)
    {
        displacements[i] = displacements[i-1] + rows_per_process[i-1];
    }

    // Finally, compute the offset and read data from the file
    matrix = new double[rows_per_process[rank] * cols];

    offset = FILE_HEADER_SIZE + displacements[rank] * cols * sizeof(double);
    MPI_File_read_at_all(file, offset, matrix, rows_per_process[rank] * cols, MPI_DOUBLE, &status);
    MPI_File_close(&file);
    return true;
}

bool utils::mpi::mpi_distributed_read_all_vector(const char* filename, double*& vector, size_t& rows, size_t& cols, int*& rows_per_process, int*& displacements)
{
    int init_flag;
    MPI_Initialized(&init_flag );
    assert(init_flag && "ERROR: MPI WAS NOT INITIALIZED.");

    int rank;
    int size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_File file;
    MPI_Offset offset;
    MPI_Status status;

    size_t header[2];

    if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS)
    {
        return false;
    }

    // First each process reads the header of the file to get the 
    MPI_File_read_at_all(file, 0, header, 2, MPI_UNSIGNED_LONG_LONG, &status);
    rows = header[0];
    cols = header[1];

    // Then the work is spread among all ranks
    rows_per_process = new int[size];
    displacements = new int[size];

    for (unsigned int i = 0; i < size; ++i)
    {
        rows_per_process[i] = rows / size;
        if (i < rows % size)
        {
            rows_per_process[i]++;
        }
    }

    displacements[0] = 0;
    for (unsigned int i = 1; i < size; ++i)
    {
        displacements[i] = displacements[i-1] + rows_per_process[i-1];
    }


    // Finally, compute the offset and read data from the file
    vector = new double[rows];
    offset = FILE_HEADER_SIZE;
    MPI_File_read_at_all(file, offset, vector, rows, MPI_DOUBLE, &status);
    MPI_File_close(&file);
    return true;
}

void utils::mpi::mpi_print_vector(const char* filename, const double* vector, const size_t rows, int* rows_per_process, int* displacements)
{
    int flag;
    MPI_Initialized(&flag);
    assert(flag && "ERROR: MPI WAS NOT INITIALIZED");

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File file;
    MPI_Status status;
    MPI_Offset offset;

    size_t cols = 1;
    size_t header[] = {rows, cols};

    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    
    // All useful data has already been computed, so we just need to write the vector to file.
    // First, start with the header (which is written by rank 0).
    if (rank == 0)
    {
        MPI_File_write(file, header, 2, MPI_UNSIGNED_LONG_LONG, &status);
    }
    // Then, considering the necessary offset, each rank writes its data.
    offset = FILE_HEADER_SIZE + displacements[rank] * sizeof(double);
    MPI_File_write_at_all(file, offset, vector + displacements[rank], rows_per_process[rank], MPI_DOUBLE, &status);

    MPI_File_close(&file);    
}
