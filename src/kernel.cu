#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <curand_kernel.h>
#include "kernel.h"

#define ROW_COUNT 6
#define COL_COUNT 7
#define WIN_COUNT 4
/// @brief Number of elements in total in Connect4 matrix
#define CONNECT4_MATRIX_SIZE (ROW_COUNT * COL_COUNT)

template <typename T> __host__ __device__ constexpr T min(T a, T b) { return (a < b) ? a : b; }
template <typename T> __host__ __device__ constexpr T max(T a, T b) { return (a > b) ? a : b; }

inline __host__ __device__  int get_flat_memory_index(int row, int col) { return ((row * COL_COUNT) + col); } 

#define IS_TABLE_VALUE_VALID(value)             ((value & 0xF0000000) == 0xC0000000)
#define SET_TABLE_VALUE_VALID(value)            value |= 0xC0000000;
#define GET_ROW_FROM_TABLE(value)               ((uint32_t)((value >> 16) & 0xFF))
#define SET_ROW_TO_TABLE_VALUE(value, row)      value |= (uint32_t)(((row & 0xFF) << 16) & 0x00FF0000)
#define GET_COL_FROM_TABLE(value)               ((uint32_t)((value >> 8) & 0xFF))
#define SET_COL_TO_TABLE_VALUE(value, col)      value |= (uint32_t)(((col & 0xFF) << 8) & 0x0000FF00)
#define GET_MOVES_FROM_TABLE(value)             ((uint32_t)(value & 0xFF))
#define SET_MOVES_TO_TABLE_VALUE(value, moves)  value |= (uint32_t)((moves & 0xFF) & 0x000000FF)

// Note: Given that the actual state is identical for all threads, best to have it communicated via 'constant' memory.
__constant__ int connect4_matrix_data[CONNECT4_MATRIX_SIZE];

__host__ cudaError_t allocate_memory(curandState** d_states, int** d_victory_table, int** d_defeat_table, int totalThreads)
{
    *d_states = nullptr;
    *d_victory_table = nullptr;
   
    auto status = cudaMalloc(d_states, sizeof(curandState) * totalThreads);
    if (status != cudaSuccess)
    {
        return status;
    }

    status = cudaMalloc(d_victory_table, sizeof(int) * totalThreads);
    if (status != cudaSuccess)
    {
        cudaFree(*d_states);
    }

    status = cudaMalloc(d_defeat_table, sizeof(int) * totalThreads);
    if (status != cudaSuccess)
    {
        cudaFree(*d_states);
        cudaFree(*d_victory_table);
    }

    return status;    
}

__host__ cudaError_t free_memory(curandState* d_states, int* d_victory_table, int* d_defeat_table)
{
    auto status = cudaFree(d_states);
    if (status != cudaSuccess)
    {
        return status;
    }

    status = cudaFree(d_defeat_table);
    if (status != cudaSuccess)
    {
        return status;
    }

    status = cudaFree(d_victory_table);
    return status;
}

__global__ void init_rand_kernel(curandState* d_states, int total_kernels, uint64_t seed)
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
    for (int col_scan = max(0, new_disc_column - (WIN_COUNT - 1)); 
         col_scan < min(COL_COUNT, new_disc_column + WIN_COUNT); 
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
    for (int row_scan = max(0, new_disc_row - (WIN_COUNT - 1));
         row_scan < min(ROW_COUNT, new_disc_row + WIN_COUNT); 
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

    for (int i = 0; i < WIN_COUNT * 2 - 1; ++i) 
    {  
        // Scan possible range
        int r = start_row + i, c = start_col + i;
        if (r < 0 || r >= ROW_COUNT || c < 0 || c >= COL_COUNT) 
            continue;

        auto current = d_matrix[get_flat_memory_index(r, c)];
        totalCountedSoFar = (current == our_disc_type) ? totalCountedSoFar + 1 : 0;

        if (totalCountedSoFar >= WIN_COUNT)
        {
            return true;
        }
    }

    // Anti-diagonal (top-right to bottom-left)
    totalCountedSoFar = 0;
    start_row = new_disc_row - min(new_disc_row, COL_COUNT - 1 - new_disc_column);
    start_col = new_disc_column + min(new_disc_row, COL_COUNT - 1 - new_disc_column);

    for (int i = 0; i < WIN_COUNT * 2 - 1; ++i) 
    {
        int r = start_row + i, c = start_col - i;
        if (r < 0 || r >= ROW_COUNT || c < 0 || c >= COL_COUNT) 
            continue;

        auto current = d_matrix[get_flat_memory_index(r, c)];
        totalCountedSoFar = (current == our_disc_type) ? totalCountedSoFar + 1 : 0;
        if (totalCountedSoFar >= WIN_COUNT) 
        {
            return true;
        }
    }
    
    // If we haven't found 4 matching entries in either row or column, then it's a no-win.
    return false;
}

__device__ inline int get_free_row_index_for_column(int* matrix, int column) // Inlining this is a good idea
{
    for (int row = ROW_COUNT - 1; row >= 0; row--)
    {
        if (matrix[get_flat_memory_index(row, column)] == 0) 
            return row;        
    }
    return -1; // No free slot here
}

__global__ void game_prediction_kernel(
    curandState* d_states, 
    int* d_victory_table, 
    int* d_defeat_table, 
    int our_disc_type)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int first_move_column = threadId % COL_COUNT;       
    extern __shared__ int shared_memory[];

    // Our slice in shared memory
    auto shared_mem_matrix = (int*)(shared_memory + (threadIdx.x * CONNECT4_MATRIX_SIZE));

    // Copy data from constant memory containing actual state to shared memory
    for (auto hover = 0; hover < CONNECT4_MATRIX_SIZE; hover++)
    {   
        shared_mem_matrix[hover] = *(((int*)connect4_matrix_data) + hover);
    }

    __syncthreads();

    // We now need to try various randomized combination 
    // with the first move being a disc inserted at column 'targetColumn'
    // Generate a floating point number between 0.0 and 1.0
    auto we_won = false;
    auto opponent_won = false;
    auto chosen_column = first_move_column;

    auto our_turn = true;
    auto opponent_disc_type = (our_disc_type == 1) ? 2 : 1;    

    auto total_our_moves = 0;
    auto total_opponent_moves = 0;

    auto opponent_winning_row = 0;
    auto opponent_winning_column = 0;
    auto prev_played_column = 0;
    auto opponent_won_on_free_column = false; 
    auto first_move_row = get_free_row_index_for_column(shared_mem_matrix, first_move_column);

    uint32_t column_occupied_flag = 0;

    for (auto col_hover = 0; col_hover < COL_COUNT; col_hover++) {
        auto occupied = get_free_row_index_for_column(shared_mem_matrix, col_hover) == -1;
        column_occupied_flag |= occupied ? (1 << col_hover) : 0;        
    }

    // Could we even play? Is the first-move column full?
    if (first_move_row != -1)
    {
        // Note: For the first run, we DO KNOW that the board is NOT full, thanks to check
        // we perform above. 
        while (column_occupied_flag != 127)
        {   
            // Any empty slots in the column?
            auto free_slot_row_index = get_free_row_index_for_column(shared_mem_matrix, chosen_column);
            if (free_slot_row_index == -1)
            {
                column_occupied_flag |= (1 << chosen_column);
                chosen_column = curand(&d_states[threadId]) % COL_COUNT;
                continue;
            }

            shared_mem_matrix[get_flat_memory_index(free_slot_row_index, chosen_column)] = our_turn ? our_disc_type : opponent_disc_type;            
            total_our_moves += our_turn ? 1 : 0;
            total_opponent_moves += !our_turn ? 1 : 0;
            
            // Do we have a win (either ours or opponents)?
            if (check_if_won(shared_mem_matrix, free_slot_row_index, chosen_column))
            {
                // Either we or the opponent has won, no need to continue the loop
                we_won = our_turn;
                opponent_won = !our_turn;
                
                if (opponent_won)
                {
                    // We need to remember this to prevent block the opponent from winning
                    opponent_winning_row = free_slot_row_index;
                    opponent_winning_column = chosen_column;
                    opponent_won_on_free_column = prev_played_column != chosen_column;
                }

                break;
            }

            // Next turn will not be ours, toggle this
            our_turn = !our_turn;
                        
            // Randomly pick next column.
            // Note: This could be further optimized to only consider empty columns, as the next random value
            // may hit a full column. The objective here is to demonstrate how the GPU could brute-force the 
            // game-board, so this optimization is omitted (among many other possible optimizations).
            prev_played_column = chosen_column;
            chosen_column = curand(&d_states[threadId]) % COL_COUNT;
        }
    }
    
    // Ensure all threads reach here first.
    __syncthreads();    

    // Update the 'd_victory_table' and 'd_defeat_table'
    if (we_won)
    {
        uint32_t value_to_store = 0;
        SET_TABLE_VALUE_VALID(value_to_store);
        SET_ROW_TO_TABLE_VALUE(value_to_store, first_move_row);
        SET_COL_TO_TABLE_VALUE(value_to_store, first_move_column);
        SET_MOVES_TO_TABLE_VALUE(value_to_store, total_our_moves);
        d_victory_table[threadId] = value_to_store;
        d_defeat_table[threadId] = -1;
    }
    else if (opponent_won && total_opponent_moves == 1 && opponent_won_on_free_column)
    {
        uint32_t value_to_store = 0;
        SET_TABLE_VALUE_VALID(value_to_store);
        SET_ROW_TO_TABLE_VALUE(value_to_store, opponent_winning_row);
        SET_COL_TO_TABLE_VALUE(value_to_store, opponent_winning_column);
        SET_MOVES_TO_TABLE_VALUE(value_to_store, total_opponent_moves);
        d_defeat_table[threadId] = value_to_store;
        d_victory_table[threadId] = -1;
    }
    else
    {
        // It's a tie (neither opponent won nor us).
        d_victory_table[threadId] = -1;
        d_defeat_table[threadId] = -1;
    }

    // Final synchronization
    __syncthreads(); 
}

bool cuda_play_turn(
    std::array<int, CONNECT4_MATRIX_SIZE> current_board_state,
    int our_disc_type,    
    int& out_best_move_row,
    int& out_best_move_column,
    bool& out_next_move_wins,
    bool& out_tie_detected)
{
    // Note: Since we have 7 columns to play as our first move,
    // we'll run 1000 threads per move, totalling to 7,000 threads
    int totalThreadsToLaunchTarget = 7000;

    // We're using shared memory to accelerate our design. Assuming the available
    // shared memory per block is 48KB, and each thread needing 168 bytes of shared-memory (6 x 7 = 42 x sizeof(int) = 168)
    // the total threads per block would be limited to 256 (although technically one could scale it up to 292 threads/block);
    int threadsPerBlock = 256;
    
    // Based on our threads per block, we calculate the total number of blocks.
    int blocksPerGrid = (totalThreadsToLaunchTarget + threadsPerBlock - 1) / threadsPerBlock;         
    int totalThreadsToLaunch = blocksPerGrid * threadsPerBlock;

    curandState* d_states;
    int* d_victory_table;
    int* d_defeat_table;

    // Allocate required memory
    auto status = allocate_memory(
        &d_states,
        &d_victory_table,
        &d_defeat_table,
        totalThreadsToLaunch);

    if (status != cudaSuccess)
    {
        printf("[cuda_play_turn] Failed to allocate memory; return code: %d\n", status);
        return false;
    }    

    // Copy data from current_board_state to connect4_matrix_data        
    status = cudaMemcpyToSymbol(connect4_matrix_data, current_board_state.data(), current_board_state.size() * sizeof(int));
    if (status != cudaSuccess)
    {
        printf("[cuda_play_turn] Failed copy matrix data to constant memory: %d\n", status);
        return false;
    }        

    // First we need to run our cuRAND initialization kernel
    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch();
    auto nano_seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    init_rand_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, totalThreadsToLaunch, nano_seed);    
    cudaDeviceSynchronize();

    // Now run the game-prediction engine
    auto shared_memory_size = threadsPerBlock * CONNECT4_MATRIX_SIZE * sizeof(int);
    game_prediction_kernel<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(d_states, d_victory_table, d_defeat_table, our_disc_type);
    cudaDeviceSynchronize();

    // We now need to copy the data from d_victory_table to out_success_per_column
    int* thread_victory_table = new int[totalThreadsToLaunch];
    int* thread_defeat_table = new int[totalThreadsToLaunch];

    cudaMemcpy(thread_victory_table, d_victory_table, sizeof(int) * totalThreadsToLaunch, cudaMemcpyDeviceToHost);
    cudaMemcpy(thread_defeat_table, d_defeat_table, sizeof(int) * totalThreadsToLaunch, cudaMemcpyDeviceToHost);

    // Gather success data
    int our_best_move_row = -1;
    int our_best_move_column = -1;
    int our_best_move_score = -1;

    int opponents_best_move_row = -1;
    int opponents_best_move_column = -1;
    int opponents_best_score = -1;

    for (auto i = 0; i < totalThreadsToLaunch; i++)
    {
        auto thread_success_valid = IS_TABLE_VALUE_VALID(thread_victory_table[i]);
        int thread_success_score = GET_MOVES_FROM_TABLE(thread_victory_table[i]);
        int thread_success_row = GET_ROW_FROM_TABLE(thread_victory_table[i]);
        int thread_success_col = GET_COL_FROM_TABLE(thread_victory_table[i]);

        auto thread_defeat_valid = IS_TABLE_VALUE_VALID(thread_defeat_table[i]);
        int thread_defeat_score =  GET_MOVES_FROM_TABLE(thread_defeat_table[i]);
        int thread_defeat_row = GET_ROW_FROM_TABLE(thread_defeat_table[i]);
        int thread_defeat_col = GET_COL_FROM_TABLE(thread_defeat_table[i]);

        // Is this the winning move?
        // Note, we're interested in lowest number of moves (the lower the score, the better)...
        if ((thread_success_valid) && ((our_best_move_score == -1) || (thread_success_score < our_best_move_score)))
        {
            our_best_move_row = thread_success_row;
            our_best_move_column = thread_success_col;
            our_best_move_score = thread_success_score;
        }

        // We also need to inspect how eay the opponent can win...
        if ((thread_defeat_valid) && ((opponents_best_score == -1) || (thread_defeat_score < opponents_best_score)))
        {
            opponents_best_score = thread_defeat_score;
            opponents_best_move_row = thread_defeat_row;
            opponents_best_move_column = thread_defeat_col;
        }
    }
    
    delete[] thread_victory_table;
    delete[] thread_defeat_table;
    
    // Now we decide whether we need to play offensive or defensive.
    // NOTE: If opponent could defeat us in 1 move, we need to play defensive and
    //       block the opponent (i.e. insert the disc to their location). Otherwise
    //       we'll make the move that gives us the highest chance to win
    out_tie_detected = false;
    out_next_move_wins = false; // Set to true of our winning score is '1'

    if (opponents_best_score == -1 && our_best_move_score == -1)
    {
        // We're a tie
        out_tie_detected = true;
    }
    else if (opponents_best_score == -1 && our_best_move_score != -1)
    {
        // We're on the offensive
        out_best_move_column = our_best_move_column;
        out_best_move_row = our_best_move_row;
        out_next_move_wins = our_best_move_score == 1; // This means we have won
    }
    else if (opponents_best_score != -1 && our_best_move_score == -1)
    {
        // We're going to lose anyway. Fire at the enemy...
        out_best_move_column = opponents_best_move_column;
        out_best_move_row = opponents_best_move_row;                
    }
    else
    {
        // We could both win and lose. If opponent wins in fewer moves than us, block them, else
        // we need to help ourselves.
        if (opponents_best_score < our_best_move_score)
        {
            out_best_move_column = opponents_best_move_column;
            out_best_move_row = opponents_best_move_row;
        }
        else
        {
            // We're on the offensive
            out_best_move_column = our_best_move_column;
            out_best_move_row = our_best_move_row;
            out_next_move_wins = our_best_move_score == 1; // This means we have won
        }
    }

    // Free memory and return
    status = free_memory(d_states, d_victory_table, d_defeat_table);
    if (status != cudaSuccess)
    {
        // We have crashed...
        printf("[cuda_play_turn] Failed to free memory, return code: %d\n", status);
        return false;
    }

    return true;
}