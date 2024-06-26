#!/bin/bash

# 5000 x 5000 per gpu
declare -A gpu_map_elems
gpu_map_elems[5000]=1
gpu_map_elems[10000]=4
gpu_map_elems[20000]=16



mkdir -p results



# Compilation
echo "Compiling the executable..."
nvcc utils.cpp main.cpp CUDA_MPI_impl.cu -I/ -lcublas -lmpi -O3

# COMMENT OUT THE CODE YOU NEED!

# Number of iterations for benchmarking
num_iter=1

# Output files
weak_scalability_elems="results/weak_scalability_elems.txt"
weak_scalability="results/weak_scalability.txt"

strong_scalability="results/strong_scalability.txt"

# Path to directory containing matrices and vectors
path_to_dir="/project/home/p200301/tests"

echo "############# WEAK SCALABILITY  OVER ELEMENTS #############"
# Loop over different matrix sizes
for matrix_size in 5000 10000 20000; do
    echo "========== Matrix Size: $matrix_size , GPU: ${gpu_map_elems[$matrix_size]} =========="
    echo "========== Matrix Size: $matrix_size , GPU: ${gpu_map_elems[$matrix_size]} ==========" >> "$weak_scalability_elems"

    matrix_file="$path_to_dir/matrix${matrix_size}.bin"
    vector_file="$path_to_dir/rhs${matrix_size}.bin"



    num_gpus="${gpu_map_elems[$matrix_size]}"

    min_time=0
    echo "Accessing directory $path_to_dir for matrix of size $matrix_size"
    
    # Calculate number of nodes
    
    # Execute program multiple times and record minimum time
    for ((iter=0; iter<$num_iter; iter++)); do
        nvcc utils.cpp main.cpp CUDA_MPI_impl.cu -I/ -lcublas -lmpi -O3
        echo Executing "srun  -n $num_gpus ./a.out $matrix_file $vector_file $matrix_size"
        time_output=$(srun -N 4 -n $num_gpus ./a.out $matrix_file $vector_file $matrix_size)
        if [ "$iter" -eq 0 ]; then
            min_time="$time_output"
        elif [ "$time_output" -lt "$min_time" ]; then
            min_time="$time_output"
        fi
    done

    # Log minimum time for the matrix size
    echo "Time: $min_time ms" >> "$weak_scalability_elems"
    echo "Time: $min_time ms"
done
echo "##########################################"

# 
declare -A gpu_map_rows
gpu_map_rows[5000]=2
gpu_map_rows[10000]=4
gpu_map_rows[20000]=8
gpu_map_rows[30000]=12
gpu_map_rows[40000]=16




echo "############# WEAK SCALABILITY OVER ROWS #############"
# Loop over different matrix sizes
for matrix_size in 5000 10000 20000 30000 40000; do
    echo "========== Matrix Size: $matrix_size , GPU: ${gpu_map_rows[$matrix_size]} =========="
    echo "========== Matrix Size: $matrix_size , GPU: ${gpu_map_rows[$matrix_size]} ==========" >> "$weak_scalability"

    matrix_file="$path_to_dir/matrix${matrix_size}.bin"
    vector_file="$path_to_dir/rhs${matrix_size}.bin"



    num_gpus="${gpu_map_rows[$matrix_size]}"

    min_time=0
    echo "Accessing directory $path_to_dir for matrix of size $matrix_size"
    
    # Calculate number of nodes
    
    # Execute program multiple times and record minimum time
    for ((iter=0; iter<$num_iter; iter++)); do
        nvcc utils.cpp main.cpp CUDA_MPI_impl.cu -I/ -lcublas -lmpi -O3
        echo Executing "srun  -n $num_gpus ./a.out $matrix_file $vector_file $matrix_size"
        time_output=$(srun -N 4 -n $num_gpus ./a.out $matrix_file $vector_file $matrix_size)
        if [ "$iter" -eq 0 ]; then
            min_time="$time_output"
        elif [ "$time_output" -lt "$min_time" ]; then
            min_time="$time_output"
        fi
    done

    # Log minimum time for the matrix size
    echo "Time: $min_time ms" >> "$weak_scalability"
    echo "Time: $min_time ms"
done
echo "##########################################"