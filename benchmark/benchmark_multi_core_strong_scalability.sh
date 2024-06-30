#!/bin/bash

mkdir -p results
mkdir -p results/multi_core
mkdir -p results/multi_core/strong

cd multi_core

# Compilation
echo "Compiling the executable..."
g++ utils.cpp main.cpp distributed.cpp -I/ -lmpi -O3


# 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1088, 1152, 1216, 1280
# Number of iterations for benchmarking
num_iter=1

# Path to directory containing matrices and vectors
path_to_dir="/project/home/p200301/tests"


# Loop over different matrix sizes
for matrix_size in 10000 20000 30000 40000 50000; do
    echo "matrix size ${matrix_size}"
    for num_proc in 4 8 16 32 64 128 256 512 1024 1088 1152 1216 1280; do
        echo "using ${num_proc} procs"
        single_time_file="../results/multi_core/strong/times${matrix_size}_${num_proc}.txt"

        matrix_file="$path_to_dir/matrix${matrix_size}.bin"
        vector_file="$path_to_dir/rhs${matrix_size}.bin"

        # Execute program multiple times and record times
        for ((iter=0; iter<$num_iter; iter++)); do
            time_output=$(srun -n $num_proc ./a.out "$matrix_file" "$vector_file" $matrix_size)
            # Trim leading and trailing whitespace
            time_output=$(echo "$time_output" | xargs)
            
            # Extract the last word
            last_substring=$(echo "$time_output" | awk '{print $NF}')
            echo "${last_substring}" >> "${single_time_file}"
            echo "run ${iter}: ${last_substring}"

        done
    done
done

echo "Finished!"
