#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include "kernel.h"

constexpr int row_count = 6;
constexpr int col_count = 7;
constexpr int win_count = 4;
constexpr int get_flat_memory_index(int row, int col) { return ((row * 7) + col); } 
template <typename T> constexpr T min(T a, T b) { return (a < b) ? a : b; }
template <typename T> constexpr T max(T a, T b) { return (a > b) ? a : b; }

/// @brief Number of elements in total in Connect4 matrix
constexpr int connect4_matrix_size = row_count * col_count;

// Note: Given that the actual state is identical for all threads, best to have
// it communicated via 'constant' memory.
__constant__ int connect4_matrix_data[42];

__host__ cudaError_t allocate_memory(curandState** d_states, int** d_success_table, int totalThreads)
{
    *d_states = nullptr;
    *d_success_table = nullptr;
   
    auto status = cudaMalloc(d_states, sizeof(curandState) * totalThreads);
    if (status != cudaSuccess)
    {
        return status;
    }

    status = cudaMalloc(d_success_table, sizeof(int) * row_count * col_count);
    if (status != cudaSuccess)
    {
        cudaFree(*d_states);
        return status;
    }
    return status;    
}

__host__ cudaError_t free_memory(curandState* d_states, int* d_success_table)
{
    auto status = cudaFree(d_states);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaFree(d_success_table);
    return status;
}

__global__ void init_rand_kernel(curandState* d_states, int total_kernels, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_kernels) {
        // Each thread gets: same seed, unique sequence (tid), no offset
        curand_init(seed, tid, 0, &d_states[tid]);
    }
}

__device__ bool check_if_won(int* d_matrix, int new_disc_row, int new_disc_column)
{
    auto our_disc_type = d_matrix[get_flat_memory_index(new_disc_row, new_disc_column)];

    // Test horizontal
    int totalCountedSoFar = 0;
    for (int col_scan = max(0, new_disc_column - (win_count - 1)); 
         col_scan < std::min(col_count, new_disc_column + win_count); 
         col_scan++)
    {
        auto currentDiscType = d_matrix[get_flat_memory_index(new_disc_row, col_scan)];
        totalCountedSoFar = (currentDiscType != our_disc_type) ? 1 : totalCountedSoFar + 1;
        if (totalCountedSoFar >= 4)
        {
            return true;
        }
    }

    // Test vertical
    totalCountedSoFar = 0;
    for (int row_scan = std::max(0, new_disc_row - (win_count - 1));
         row_scan < std::min(row_count, new_disc_row + win_count); 
         row_scan++)
    {
        auto currentDiscType = d_matrix[get_flat_memory_index(row_scan, new_disc_column)];
        totalCountedSoFar = (currentDiscType != our_disc_type) ? 1 : totalCountedSoFar + 1;
        if (totalCountedSoFar >= 4)
        {
            return true;
        }
    }
    
    // If we haven't found 4 matching entries in either row or column, then it's a no-win.
    return false;
}

__device__ int get_free_row_index_for_column(int* d_matrix, int column)
{
    for (int row = row_count - 1; row >= 0; row--)
    {
        if (d_matrix[get_flat_memory_index(row, column)] == 0) 
            return row;        
    }
    return -1; // No free slot here
}

__global__ void game_prediction_kernel(
    curandState* d_states, 
    int* d_success_table, 
    int our_disc_id)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIdInBlock = threadIdx.x;
    int targetColumn = threadId % col_count;    
    int successCount = 0;
    extern __shared__ int shared_memory[];

    // Save the state of the randomizer
    auto saved_curand_state = d_states[threadId];

    // Our slice in shared memory
    int* local_shared_mem = (int*)(shared_memory + threadIdInBlock);    

    // Copy data from constant memory containing actual state to shared memory
    for (auto hover = 0; hover < connect4_matrix_size; hover++)
    {   
        local_shared_mem[hover] = *(((int*)connect4_matrix_data) + hover);
    }
    __syncthreads();

    // We now need to try various randomized combination 
    // with the first move being a disc inserted at column 'targetColumn'



    // Ensure all threads reach here first.
    __syncthreads();    

    // Restore randomizer state for the next kernel call. Also update the 'd_success_table'
    d_states[threadId] = saved_curand_state; 
    atomicAdd(&d_success_table[targetColumn], successCount);
}

__host__ bool play_game(
    std::vector<std::vector<int>> current_board_state,
    int our_disc_type,
    std::vector<int>& out_success_per_column,
    int& out_best_move_column,
    bool& out_next_move_wins)
{
    // Note: Since we have 7 columns to play as our first move,
    // we'll run 1000 threads per move, totalling to 7,000 threads
    int totalThreadsToLaunch = 7000;

    // We're using shared memory to accelerate our design. Assuming the available
    // shared memory per block is 48KB, and each thread needing 168 bytes of shared-memory (6 x 7 = 42 x sizeof(int) = 168)
    // the total threads per block would be limited to 256 (although technically one could scale it up to 292 threads/block);
    int threadsPerBlock = 256;
    
    // Based on our threads per block, we calculate the total number of blocks.
    int blocksPerGrid = (totalThreadsToLaunch + threadsPerBlock - 1) / threadsPerBlock;

    int matrix_buffer[row_count * col_count];
    int success_per_column[] = {0, 0, 0, 0, 0, 0, 0};

    curandState* d_states;
    int* d_success_table;
    
    // Allocate required memory
    auto status = allocate_memory(&d_states, &d_success_table, totalThreadsToLaunch);
    if (status != cudaSuccess)
    {
        printf("Failed to allocate memory; return code: %d\n", status);
        return false;
    }

    // First we need to run our cuRAND initialization kernel
    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch();
    unsigned long long nano_seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    init_rand_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, totalThreadsToLaunch, nano_seed);    
    cudaDeviceSynchronize();

    // Now run the game-prediction engine
    game_prediction_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, d_success_table, our_disc_type);
    cudaDeviceSynchronize();

    // Free memory and return
    status = free_memory(d_states, d_success_table);
    if (status != cudaSuccess)
    {
        // We have crashed...
        printf("Failed to free memory, return code: %d\n", status);
        return false;
    }
    return true;
}
