#include "strat.hpp"

    double dot(const double * x, const double * y, size_t size) 
    {
        double result = 0.0;
        for(size_t i = 0; i < size; i++)
        {
            result += x[i] * y[i];
        }
        return result;
    }



    void axpby(double alpha, const double * x, double beta, double * y, size_t size) 
    {
        // y = alpha * x + beta * y

        for(size_t i = 0; i < size; i++)
        {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }



    void gemv(double alpha, const double * A, const double * x, double beta, double *& y, size_t num_rows, size_t num_cols, int displ) 
    {
        // y = alpha * A * x + beta * y;
        // int rank;

        // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // if (rank == 1)
        // {
        //     std::cout << num_rows << "aaaaaaaaaa" << displ << std::endl;
        //     for (int j = 0; j < num_cols; ++j)
        //     {
        //         std::cout << x[j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        for(size_t r = 0; r < num_rows; r++)
        {
            double y_val = 0.0;
            for(size_t c = 0; c < num_cols; c++)
            {
                // if (rank == 1){
                //     std::cout << "item: "  << r * num_cols + c << std::endl;
                //     std::cout << "matrix element " << A[r * num_cols + c] << std::endl;
                //     std::cout << "vector element " << x[c] << std::endl;
                // }
                y_val += alpha * A[(r) * num_cols + c] * x[c];

            }
            y[r + displ] = y_val;
            // std::cout << "--->" << y[r + displ] << std::endl;
        }
    }


    /**
     * 
    */ 
    void conjugate_gradient(const double * A, const double * b, double * x, size_t rows, size_t cols, int* rows_per_process, int* displacements, int max_iters, double rel_error) 
    {
        double alpha, beta, bb, rr, rr_new;
        double * r = new double[rows];
        double * p = new double[rows];
        double * Ap = new double[rows];
        int num_iters;

        int size;
        int rank;

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        memset(x, 0, rows * sizeof(double));
        memcpy(r, b, rows * sizeof(double));
        memcpy(p, b, rows * sizeof(double));

        
        MPI_Barrier(MPI_COMM_WORLD);

        bb = dot(b, b, rows);
        rr = bb;
        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {
            gemv(1.0, A, p, 0.0, Ap, rows_per_process[rank], rows, displacements[rank]);
            
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Allgatherv(Ap + displacements[rank], rows_per_process[rank], MPI_DOUBLE, Ap, rows_per_process, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
            
            alpha = rr / dot(p, Ap, rows);
            axpby(alpha, p, 1.0, x, rows);
            axpby(-alpha, Ap, 1.0, r, rows);
            rr_new = dot(r, r, rows);

            beta = rr_new / rr;
            rr = rr_new;

            if(std::sqrt(rr / bb) < rel_error) { break; }
            axpby(1.0, r, beta, p, rows);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        delete[] r;
        delete[] p;
        delete[] Ap;

        if(num_iters <= max_iters && rank == 0)  
        {
            //printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        }
        else if (num_iters > max_iters)
        {
            //printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
        }
        
    }
    
