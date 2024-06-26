#!/bin/bash

mkdir -p results
mkdir -p results/single_gpu

cd results/single_gpu

# Compilation
echo "Compiling the executable..."
nvcc CudaSingleGPU.cpp CUBLAS_single.cu utils.cpp -I/ -lcublas -lmpi -O3

# Number of iterations for benchmarking
num_iter=30

# Path to directory containing matrices and vectors
path_to_dir="/project/home/p200301/tests/"

echo "========== Average times over $num_iter runs =========="
for matrix_size in 100 500 1000 5000 10000 20000 30000 40000 50000 60000 70000; do
    matrix_file="$path_to_dir/matrix${matrix_size}.bin"
    vector_file="$path_to_dir/rhs${matrix_size}.bin"
    times_file="../results/single_gpu/times${matrix_size}.txt"

    echo "Accessing directory $path_to_dir for matrix of size $matrix_size"
    
    
    # Execute program multiple times and collect timings
    for ((iter=0; iter<$num_iter; iter++)); do
        time_output=$(./a.out "$matrix_file" "$vector_file" $matrix_size)

        echo "run $iter: $time_output"
        echo "$time_output" >> "$times_file"
    done
done

echo Finished!
rm a.out
