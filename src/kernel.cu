#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <algorithm>
#include "kernel.h"

#define row_count 6
#define col_count 7
#define win_count 4
#define connect4_matrix_size (row_count * col_count)
inline __device__ constexpr int get_flat_memory_index(int row, int col) { return ((row * 7) + col); } 
template <typename T> constexpr T min(T a, T b) { return (a < b) ? a : b; }
template <typename T> constexpr T max(T a, T b) { return (a > b) ? a : b; }

/// @brief Number of elements in total in Connect4 matrix

// Note: Given that the actual state is identical for all threads, best to have
// it communicated via 'constant' memory.
__constant__ int connect4_matrix_data[connect4_matrix_size];

__host__ cudaError_t allocate_memory(curandState** d_states, int** d_success_table, int totalThreads)
{
    *d_states = nullptr;
    *d_success_table = nullptr;
   
    auto status = cudaMalloc(d_states, sizeof(curandState) * totalThreads);
    if (status != cudaSuccess)
    {
        return status;
    }

    status = cudaMalloc(d_success_table, sizeof(int) * col_count);
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
        totalCountedSoFar = (currentDiscType != our_disc_type) ? 0 : totalCountedSoFar + 1;
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
        totalCountedSoFar = (currentDiscType != our_disc_type) ? 0 : totalCountedSoFar + 1;
        if (totalCountedSoFar >= 4)
        {
            return true;
        }
    }

    // Diagonal (top-left to bottom-right)
    totalCountedSoFar = 0;
    int start_row = new_disc_row - min(new_disc_row, new_disc_column);
    int start_col = new_disc_column - min(new_disc_row, new_disc_column);

    for (int i = 0; i < win_count * 2 - 1; ++i) 
    {  
        // Scan possible range
        int r = start_row + i, c = start_col + i;
        if (r < 0 || r >= row_count || c < 0 || c >= col_count) 
            continue;

        auto current = d_matrix[get_flat_memory_index(r, c)];
        totalCountedSoFar = (current == our_disc_type) ? totalCountedSoFar + 1 : 0;

        if (totalCountedSoFar >= win_count)
        {
            return true;
        }
    }

    // Anti-diagonal (top-right to bottom-left)
    totalCountedSoFar = 0;
    start_row = new_disc_row - min(new_disc_row, col_count - 1 - new_disc_column);
    start_col = new_disc_column + min(new_disc_row, col_count - 1 - new_disc_column);

    for (int i = 0; i < win_count * 2 - 1; ++i) 
    {
        int r = start_row + i, c = start_col - i;
        if (r < 0 || r >= row_count || c < 0 || c >= col_count) 
            continue;

        auto current = d_matrix[get_flat_memory_index(r, c)];
        totalCountedSoFar = (current == our_disc_type) ? totalCountedSoFar + 1 : 0;
        if (totalCountedSoFar >= win_count) 
        {
            return true;
        }
    }
    
    // If we haven't found 4 matching entries in either row or column, then it's a no-win.
    return false;
}

__device__ inline int get_free_row_index_for_column(int* d_matrix, int column) // Inlining this is a good idea
{
    for (int row = row_count - 1; row >= 0; row--)
    {
        if (d_matrix[get_flat_memory_index(row, column)] == 0) 
            return row;        
    }
    return -1; // No free slot here
}

__device__ inline bool is_column_full(int* d_matrix, int column)
{
    // Note: Row 0 is the top-most row, and Row 6 is the bottom most.
    // When you insert a disc, it falls to the bottom-most position. Hence if top-most
    // rows is full, it means the whole column is full.
    return d_matrix[get_flat_memory_index(0, column)] != 0;
}

__global__ void game_prediction_kernel(
    curandState* d_states, 
    int* d_success_table, 
    int our_disc_type)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIdInBlock = threadIdx.x;
    int first_move_column = threadId % col_count;       
    extern __shared__ int shared_memory[];

    // Our slice in shared memory
    auto shared_mem_matrix = (int*)(shared_memory + (threadIdInBlock * connect4_matrix_size));

    // Copy data from constant memory containing actual state to shared memory
    for (auto hover = 0; hover < connect4_matrix_size; hover++)
    {   
        shared_mem_matrix[hover] = *(((int*)connect4_matrix_data) + hover);
    }
    __syncthreads();

    // We now need to try various randomized combination 
    // with the first move being a disc inserted at column 'targetColumn'
    // Generate a floating point number between 0.0 and 1.0
    auto game_won = false;
    auto chosen_column = first_move_column;

    auto our_turn = true;
    auto opponent_disc_type = (our_disc_type == 1) ? 2 : 1;

    auto total_available_slots = 0;
    for (auto col_hover = 0; col_hover < col_count; col_hover++)
    {
        total_available_slots += (get_free_row_index_for_column(shared_mem_matrix, col_hover) + 1);
    }

    // Could we even play? Is the first-move column full?
    if (get_free_row_index_for_column(shared_mem_matrix, first_move_column) != -1)
    {
        // Note: For the first run, we DO KNOW that the board is NOT full, thanks to check
        // we perform above. 
        while (total_available_slots > 0)
        {   
            // Any empty slots in the column?
            auto free_slot_row_index = get_free_row_index_for_column(shared_mem_matrix, chosen_column);
            if (free_slot_row_index == -1)
            {                
                chosen_column = curand(&d_states[threadId]) % col_count;
                continue;
            }

            shared_mem_matrix[get_flat_memory_index(free_slot_row_index, chosen_column)] = our_turn ? our_disc_type : opponent_disc_type;
            
            // Do we have a win (either ours or opponents)?
            if (check_if_won(shared_mem_matrix, free_slot_row_index, chosen_column))
            {
                // Either we or the opponent has won, no need to continue the loop
                game_won = our_turn ? true : false; 
                break;
            }

            // Next turn will not be ours, toggle this
            our_turn = !our_turn;
                        
            // Randomly pick next column.
            // Note: This could be further optimized to only consider empty columns, as the next random value
            // may hit a full column. The objective here is to demonstrate how the GPU could brute-force the 
            // game-board, so this optimization is omitted (among many other possible optimizations).
            chosen_column = curand(&d_states[threadId]) % col_count;

            // One more slot was occupied, add this
            total_available_slots--;
        }
    }
    
    // Ensure all threads reach here first.
    __syncthreads();    

    // Restore randomizer state for the next kernel call. Also update the 'd_success_table'
    atomicAdd(&d_success_table[first_move_column], game_won ? 1 : 0);

    // Final synchronization
    __syncthreads(); 
}

__host__ bool play_game(
    std::array<int, 42> current_board_state,
    int our_disc_type,
    std::array<int, 7>& out_success_per_column,
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

    curandState* d_states;
    int* d_success_table;
    
    // Allocate required memory
    auto status = allocate_memory(&d_states, &d_success_table, totalThreadsToLaunch);
    if (status != cudaSuccess)
    {
        printf("Failed to allocate memory; return code: %d\n", status);
        return false;
    }    

    // Copy data from current_board_state to connect4_matrix_data        
    status = cudaMemcpyToSymbol(connect4_matrix_data, current_board_state.data(), current_board_state.size() * sizeof(int));
    if (status != cudaSuccess)
    {
        printf("Failed copy matrix data to constant memory: %d\n", status);
        return false;
    }        

    // First we need to run our cuRAND initialization kernel
    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch();
    unsigned long long nano_seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();    

    init_rand_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, totalThreadsToLaunch, nano_seed);    
    cudaDeviceSynchronize();

    // Now run the game-prediction engine
    auto shared_memory_size = threadsPerBlock * connect4_matrix_size * sizeof(int);
    game_prediction_kernel<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(d_states, d_success_table, our_disc_type);
    cudaDeviceSynchronize();

    // We now need to copy the data from d_success_table to out_success_per_column
    cudaMemcpy(out_success_per_column.data(), d_success_table, sizeof(int) * col_count, cudaMemcpyDeviceToHost);

    // Gather success data
    int highest_score = 0;
    int highest_score_column_index = -1;
    for (auto i = 0; i < out_success_per_column.size(); i++)
    {
        if (out_success_per_column[i] > highest_score)
        {
            highest_score_column_index = i;
            highest_score = out_success_per_column[i];
        }
    }

    out_best_move_column = highest_score_column_index;
    out_next_move_wins = highest_score > 0; // The score indicates the number of potential wins for the next move, with '0' indicating a no-win prediction.
    
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
