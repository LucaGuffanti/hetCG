#!/bin/bash

# 5000 x 5000 per gpu
declare -A gpu_map_elems
gpu_map_elems[5000]=1
gpu_map_elems[10000]=4
gpu_map_elems[20000]=16



mkdir -p results
mkdir -p results/multi_gpu
mkdir -p results/multi_gpu/weak

cd multi_gpu

# Compilation
echo "Compiling the executable..."
nvcc utils.cpp main.cpp CUDA_MPI_impl.cu -I/ -lcublas -lmpi -O3

# Number of iterations for benchmarking
num_iter=30

# Output files
averages="../results/multi_gpu/weak/averages.txt"

# Path to directory containing matrices and vectors
path_to_dir="/project/home/p200301/tests"

# Loop over different matrix sizes
for matrix_size in 5000 10000 20000; do
    matrix_file="$path_to_dir/matrix${matrix_size}.bin"
    vector_file="$path_to_dir/rhs${matrix_size}.bin"

    num_gpus="${gpu_map_elems[$matrix_size]}"
    times_file="../results/multi_gpu/weak/times${matrix_size}.txt"

    echo "Accessing directory $path_to_dir for matrix of size $matrix_size"
    
    # Calculate number of nodes
    
    # Execute program multiple times and record minimum time
    for ((iter=0; iter<$num_iter; iter++)); do
        nvcc utils.cpp main.cpp CUDA_MPI_impl.cu -I/ -lcublas -lmpi -O3
        time_output=$(srun -N 4 -n $num_gpus ./a.out $matrix_file $vector_file $matrix_size)

        last_substring="${time_output##* }"
        echo "run ${iter}: ${last_substring}"
        echo "${last_substring}" >> "${times_file}"
    done

    # Log minimum time for the matrix size
    echo "Time: $min_time ms" >> "$weak_scalability_elems"
    echo "Time: $min_time ms"
done

echo "Finished!