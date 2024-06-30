#!/bin/bash

# 5000 x 5000 per gpu
declare -A processes

processes[5000]=1
processes[10000]=4
processes[20000]=16
processes[30000]=36
processes[40000]=64
processes[50000]=100




mkdir -p results



# Compilation
echo "Compiling the executable..."
g++ utils.cpp main.cpp distributed.cpp -I/ -lmpi -O3

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
for matrix_size in 5000 10000 20000 30000 40000 50000; do
    echo "========== Matrix Size: $matrix_size , MPI PROCESSES: ${processes[$matrix_size]} =========="
    echo "========== Matrix Size: $matrix_size , MPI PROCESSES: ${processes[$matrix_size]} ==========" >> "$weak_scalability_elems"

    matrix_file="$path_to_dir/matrix${matrix_size}.bin"
    vector_file="$path_to_dir/rhs${matrix_size}.bin"



    num_procs="${processes[$matrix_size]}"

    min_time=0
    echo "Accessing directory $path_to_dir for matrix of size $matrix_size"
    
    # Calculate number of nodes
    
    # Execute program multiple times and record minimum time
    for ((iter=0; iter<$num_iter; iter++)); do
        g++ utils.cpp main.cpp distributed.cpp -I/ -lmpi -O3

        echo Executing "srun  -n $num_procs ./a.out $matrix_file $vector_file"
        time_output=$(srun -n $num_procs ./a.out $matrix_file $vector_file)
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


declare -A processes_map_rows
processes_map_rows[5000]=2
processes_map_rows[10000]=4
processes_map_rows[20000]=8
processes_map_rows[30000]=12
processes_map_rows[40000]=16




echo "############# WEAK SCALABILITY OVER ROWS #############"
# Loop over different matrix sizes
for matrix_size in 5000 10000 20000 30000 40000; do
    echo "========== Matrix Size: $matrix_size , MPI PROCESSES: ${processes_map_rows[$matrix_size]} =========="
    echo "========== Matrix Size: $matrix_size , MPI PROCESSES: ${processes_map_rows[$matrix_size]} ==========" >> "$weak_scalability"

    matrix_file="$path_to_dir/matrix${matrix_size}.bin"
    vector_file="$path_to_dir/rhs${matrix_size}.bin"



    num_procs="${processes_map_rows[$matrix_size]}"

    min_time=0
    echo "Accessing directory $path_to_dir for matrix of size $matrix_size"
    
    # Calculate number of nodes
    
    # Execute program multiple times and record minimum time
    for ((iter=0; iter<$num_iter; iter++)); do
           g++ utils.cpp main.cpp distributed.cpp -I/ -lmpi -O3
        
        echo Executing "srun  -n $num_procs ./a.out $matrix_file $vector_file $matrix_size"
        time_output=$(srun -N 4 -n $num_procs ./a.out $matrix_file $vector_file $matrix_size)
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

echo "############# STRONG SCALABILITY ############"
matrix_size=50000
# Loop over different matrix sizes
for ((num_procs=4; num_procs<=1024; num_procs=num_procs*2)); do
    echo "========== PROCESSES: $num_procs =========="
    echo "========== PROCESSES: $num_procs ==========" >> "$strong_scalability"

    matrix_file="$path_to_dir/matrix${matrix_size}.bin"
    vector_file="$path_to_dir/rhs${matrix_size}.bin"

    min_time=0
    echo "Accessing directory $path_to_dir for matrix of size $matrix_size"
    
    # Execute program multiple times and record minimum time
    for ((iter=0; iter<=$num_iter; iter++)); do
        echo Executing "srun  -n $num_procs ./a.out $matrix_file $vector_file"
        g++ utils.cpp main.cpp distributed.cpp -I/ -lmpi -O3
        time_output=$(srun -n $num_procs ./a.out "$matrix_file" "$vector_file")
        last_substring="${time_output##* }"
        min_time="$last_substring"
    done

    # Log minimum time for the matrix size
    echo "Minimum Time: $min_time ms" >> "$strong_scalability"
    echo "Minimum Time: $min_time ms"
done
echo "##########################################"

# echo Finished!
