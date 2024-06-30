#!/bin/bash

mkdir -p results
mkdir -p results/multi_gpu
mkdir -p results/multi_gpu/strong

cd multi_gpu


# Compilation
echo "Compiling the executable..."
nvcc utils.cpp main.cpp CUDA_MPI_impl.cu -I/ -lcublas -lmpi -O3

# Number of iterations for benchmarking
NUM_ITER=30
MATRIX_SIZE=70000


# Path to directory containing matrices and vectors
path_to_dir="/project/home/p200301/tests"

# Output file for averages
averages_file="../results/multi_gpu/strong/averages.txt"
echo "matrix_size gpus time" >> "${averages_file}"

# Loop over different matrix sizes

echo "Matrix size: ${MATRIX_SIZE}"

for ((num_gpus=4; num_gpus<=16; num_gpus=num_gpus+1)); do
    echo "using ${num_gpus} gpus"
    single_time_file="../results/multi_gpu/strong/times${MATRIX_SIZE}_${num_gpus}.txt"

    matrix_file="$path_to_dir/matrix${MATRIX_SIZE}.bin"
    vector_file="$path_to_dir/rhs${MATRIX_SIZE}.bin"

    # Execute program multiple times and record times
    for ((iter=0; iter<$NUM_ITER; iter++)); do
        time_output=$(srun -N 4 -n $num_gpus ./a.out "$matrix_file" "$vector_file" $MATRIX_SIZE)
        last_substring="${time_output##* }"
        echo "${last_substring}" >> "${single_time_file}"
        echo "run ${iter}: ${last_substring}"

    done

    # Process times to calculate the average after removing max and min
    times=($(sort -n "${single_time_file}"))
    total_time=0
    count=0
    for ((i=1; i<${#times[@]}-1; i++)); do
        total_time=$(echo "$total_time + ${times[$i]}" | bc)
        count=$((count + 1))
    done

    if [ $count -gt 0 ]; then
        avg_time=$(echo "scale=6; $total_time / $count" | bc)
        echo "${MATRIX_SIZE} ${num_gpus} ${avg_time}" >> "${averages_file}"
    fi

done

echo "Finished!"
