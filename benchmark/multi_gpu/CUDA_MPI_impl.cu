#include "CUDA_MPI_header.hpp"


#define cublasCheckError(status) \
    { \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("CUBLAS Error: %d\n", status); \
            cudaDeviceReset(); \
            exit(1); \
        } \
    }

static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}


void conjugate_gradient(const double* distr_A,
  double* x, 
  double* b, 
  const unsigned int rows, 
  const unsigned int cols, 
  const int* rows_per_process, 
  const int* displacements, 
  const int max_iter, 
  const double rel_err)
  {
    int flag;
    MPI_Initialized(&flag);
    assert(flag && "ERROR: MPI WAS NOT INITIALIZED");
    
    int rank;
    int mpi_size;
    int local_gpu_rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    //calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[mpi_size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
    
    for (int p=0; p<mpi_size; p++) {
      if (p == rank) break;
      if (hostHashs[p] == hostHashs[rank]) local_gpu_rank++;
    }
    
    cublasHandle_t handle;
    
    // Choose the GPU based on local rank
    cudaSetDevice(local_gpu_rank);
    // Initialize NCCL
    cublasCreate(&handle);
    
    const int m = rows_per_process[rank];
    const int n = cols;
    
    double alpha, beta, bb, rr, rr_new;
    int num_iters;
    
    
    // Vectors used on the GPU
    double* dev_loc_A;
    double* dev_x;
    double* dev_b;
    double* dev_r;
    double* dev_p;
    double* dev_Ap;
    double* dev_loc_Ap;

    double* host_Ap = (double*) malloc(sizeof(double) * cols);
    double* local_Ap = (double*) malloc(sizeof(double) * rows_per_process[rank]);
    // Allocating vectors on GPU
    // std::cout << "Rank " << rank << " will do " << m << " rows" << std::endl;
    
    cudaMalloc(&dev_loc_A, rows_per_process[rank] * cols * sizeof(double));
    cudaMalloc(&dev_x, rows * sizeof(double));
    cudaMalloc(&dev_b, rows * sizeof(double));
    cudaMalloc(&dev_r, rows * sizeof(double));
    cudaMalloc(&dev_p, rows * sizeof(double));
    cudaMalloc(&dev_Ap, rows * sizeof(double));
    cudaMalloc(&dev_loc_Ap, rows_per_process[rank] * sizeof(double));

    
    // Copying data from CPU to GPU
    cudaMemcpy(dev_loc_A, distr_A, rows_per_process[rank] * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(dev_x, 0, rows * sizeof(double));
    cudaMemcpy(dev_b, b, rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r, dev_b, rows * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_p, dev_b, rows * sizeof(double), cudaMemcpyDeviceToDevice);

    double one = 1.0;
    double zero = 0.0;
    double alpha2;
    double den;
    
    cublasDdot(handle, rows, dev_b, 1, dev_b, 1, &bb);
    
    rr = bb;

    MPI_Barrier(MPI_COMM_WORLD);
    for(num_iters = 1; num_iters <= max_iter; num_iters++)
    {
      
      // Compute the gemv
      cublasCheckError(cublasDgemv(handle, CUBLAS_OP_T, n, m, &one, dev_loc_A, n, dev_p, 1, &zero, dev_loc_Ap, 1));
      cudaMemcpy(local_Ap, dev_loc_Ap, rows_per_process[rank] * sizeof(double), cudaMemcpyDeviceToHost);
      
    
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allgatherv(local_Ap, rows_per_process[rank], MPI_DOUBLE, host_Ap, rows_per_process, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
      cudaMemcpy(dev_Ap, host_Ap, cols * sizeof(double), cudaMemcpyHostToDevice);

      
      cublasDdot(handle, rows, dev_p, 1, dev_Ap, 1, &den);
      
      alpha = rr / den;
      
      // axpby(alpha, p, 1.0, x, size);
      cublasDaxpy(handle, rows, &alpha, dev_p, 1, dev_x, 1);
      
      // axpby(-alpha, Ap, 1.0, r, size);
      alpha2 = -alpha;
      cublasDaxpy(handle, rows, &alpha2, dev_Ap, 1, dev_r, 1);
      cublasDdot(handle, rows, dev_r, 1, dev_r, 1, &rr_new);
      
      beta = rr_new / rr;
      rr = rr_new;
      
      if(std::sqrt(rr / bb) < rel_err) { break; }
      cublasDscal(handle, rows, &beta, dev_p, 1);
      cublasDaxpy(handle, rows, &one, dev_r, 1, dev_p, 1);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if(num_iters <= max_iter && rank == 0)
    {
      //printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
      cudaMemcpy(x, dev_x, cols * sizeof(double), cudaMemcpyDeviceToHost);
    }
    else if (num_iters > max_iter)
    {
      printf("Did not converge in %d iterations, relative error is %e\n", max_iter, std::sqrt(rr / bb));
    }

    cudaFree(dev_loc_A);
    cudaFree(dev_x);
    cudaFree(dev_b);
    cudaFree(dev_r);
    cudaFree(dev_p);
    cudaFree(dev_Ap);

    delete [] host_Ap;
    delete [] local_Ap;
  } 
  