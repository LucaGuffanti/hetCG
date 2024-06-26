#include "CUBLAS_single.hpp"


__host__ void conjugate_gradient_blas(
    double* A, 
    double* x, 
    double* b, 
    int size,
    int max_iters, 
    double rel_error
)
{
    cublasHandle_t cublasH = NULL;

    cublasCreate(&cublasH);

    const int m = size;
    const int n = size;

    

    double alpha, beta, bb, rr, rr_new;
    int num_iters;


    // Vectors used on the GPU
    double* dev_A;
    double* dev_x;
    double* dev_b;
    double* dev_r;
    double* dev_p;
    double* dev_Ap;

    // Allocating vectors on GPU
    const unsigned int matrix_bytes = m * n * sizeof(double);
    const unsigned int vector_bytes = m * sizeof(double);

    cudaMalloc(&dev_A, matrix_bytes);
    cudaMalloc(&dev_x, vector_bytes);
    cudaMalloc(&dev_b, vector_bytes);
    cudaMalloc(&dev_r, vector_bytes);
    cudaMalloc(&dev_p, vector_bytes);
    cudaMalloc(&dev_Ap, vector_bytes);

    // Copying data from CPU to GPU
    cudaMemcpy(dev_A, A, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemset(dev_x, 0, vector_bytes);
    cudaMemcpy(dev_b, b, vector_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r, dev_b, vector_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_p, dev_b, vector_bytes, cudaMemcpyDeviceToDevice);


    // bb = dot(b, b, size);
    // Compute the dot product
    double one = 1.0;
    double zero = 0.0;
    double alpha2;
    double den;

    cublasDdot(cublasH, m, dev_b, 1, dev_b, 1, &bb);
    
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {

        cublasDgemv(cublasH, CUBLAS_OP_N, m, n, &one, dev_A, m, dev_p, 1, &zero, dev_Ap, 1);
        cublasDdot(cublasH, m, dev_p, 1, dev_Ap, 1, &den);
        alpha = rr / den;

        // axpby(alpha, p, 1.0, x, size);
        cublasDaxpy(cublasH, m, &alpha, dev_p, 1, dev_x, 1);
        

        // axpby(-alpha, Ap, 1.0, r, size);
        alpha2 = -alpha;
        cublasDaxpy(cublasH, m, &alpha2, dev_Ap, 1, dev_r, 1);

        cublasDdot(cublasH, m, dev_r, 1, dev_r, 1, &rr_new);

        beta = rr_new / rr;
        rr = rr_new;

        if(std::sqrt(rr / bb) < rel_error) { break; }
        // axpby(1.0, r, beta, p, size);
        cublasDscal(cublasH, m, &beta, dev_p, 1);
        cublasDaxpy(cublasH, m, &one, dev_r, 1, dev_p, 1);
    }

    if(num_iters <= max_iters)
    {
        //printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        cudaMemcpy(x, dev_x, vector_bytes, cudaMemcpyDeviceToHost);
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }

    cudaFree(dev_A);
    cudaFree(dev_x);
    cudaFree(dev_b);
    cudaFree(dev_r);
    cudaFree(dev_p);
    cudaFree(dev_Ap);
}