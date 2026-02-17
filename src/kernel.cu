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

// Note: Given that the actual state is identical for all threads, best to have
// it communicated via 'constant' memory.
__constant__ int connect4_matrix_data[CONNECT4_MATRIX_SIZE];

__host__ cudaError_t allocate_memory(curandState** d_states, int** d_success_table, int totalThreads)
{
    *d_states = nullptr;
    *d_success_table = nullptr;
   
    auto status = cudaMalloc(d_states, sizeof(curandState) * totalThreads);
    if (status != cudaSuccess)
    {
        return status;
    }

    status = cudaMalloc(d_success_table, sizeof(int) * totalThreads);
    if (status != cudaSuccess)
    {
        cudaFree(*d_states);
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

__host__ __device__ inline int get_free_row_index_for_column(int* matrix, int column) // Inlining this is a good idea
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
    int* d_success_table, 
    int our_disc_type)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIdInBlock = threadIdx.x;
    int first_move_column = threadId % COL_COUNT;       
    extern __shared__ int shared_memory[];

    if (threadId == 0) {  // limit output to one print per block
        DEBUG(printf("[kernel::game_prediction_kernel] Kernel launched: block %d  |  our_disc_type = %d  |  threads = %d\n", blockIdx.x, our_disc_type, blockDim.x))
    }

    // Our slice in shared memory
    auto shared_mem_matrix = (int*)(shared_memory + (threadIdInBlock * CONNECT4_MATRIX_SIZE));

    // Copy data from constant memory containing actual state to shared memory
    for (auto hover = 0; hover < CONNECT4_MATRIX_SIZE; hover++)
    {   
        shared_mem_matrix[hover] = *(((int*)connect4_matrix_data) + hover);
    }
    __syncthreads();

    if (threadId == 0) {  // limit output to one print per block
        DEBUG(printf("[kernel::game_prediction_kernel] Shared data copied\n"))
    }

    // We now need to try various randomized combination 
    // with the first move being a disc inserted at column 'targetColumn'
    // Generate a floating point number between 0.0 and 1.0
    auto game_won = false;
    auto chosen_column = first_move_column;

    auto our_turn = true;
    auto opponent_disc_type = (our_disc_type == 1) ? 2 : 1;
    auto total_available_slots = 0;
    auto total_moves_to_success = 0;

    for (auto col_hover = 0; col_hover < COL_COUNT; col_hover++)
    {
        total_available_slots += (get_free_row_index_for_column(shared_mem_matrix, col_hover) + 1);
    }

    if (threadId == 0) {  // limit output to one print per block
        DEBUG(printf("[kernel::game_prediction_kernel] Total available slots calculated to %d, launching game loop... \n", total_available_slots))
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
                chosen_column = curand(&d_states[threadId]) % COL_COUNT;
                continue;
            }

            shared_mem_matrix[get_flat_memory_index(free_slot_row_index, chosen_column)] = our_turn ? our_disc_type : opponent_disc_type;
            total_moves_to_success++;
            
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
            chosen_column = curand(&d_states[threadId]) % COL_COUNT;

            // One more slot was occupied, add this
            total_available_slots--;
        }
    }

    if (threadId == 0) {  // limit output to one print per block
        DEBUG(printf("[kernel::game_prediction_kernel] Game loop complete\n"))
    }
    
    // Ensure all threads reach here first.
    __syncthreads();    

    // Restore randomizer state for the next kernel call. Also update the 'd_success_table'

    if (threadId > 7167 ) {
        DEBUG(printf("[kernel::game_prediction_kernel] Illegal thread ID detected: %d.\n", threadId))
    }

    d_success_table[threadId] = game_won ? total_moves_to_success : -1;

    if (threadId == 0) {  // limit output to one print per block
        DEBUG(printf("[kernel::game_prediction_kernel] updated d_success_table, finishing...\n"))
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
    DEBUG(std::cout << "[cuda_play_turn] entering 'cuda_play_turn'... " << std::endl)
    
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
    int* d_success_table;
    
    DEBUG(std::cout << "[cuda_play_turn] Attempting to allocate device memory... " << std::endl)

    // Allocate required memory
    auto status = allocate_memory(&d_states, &d_success_table, totalThreadsToLaunch);
    if (status != cudaSuccess)
    {
        DEBUG(printf("[cuda_play_turn] Failed to allocate memory; return code: %d\n", status))
        return false;
    }    

    DEBUG(std::cout << "[cuda_play_turn] Attempting to copy data to constant memory ... " << std::endl)

    // Copy data from current_board_state to connect4_matrix_data        
    status = cudaMemcpyToSymbol(connect4_matrix_data, current_board_state.data(), current_board_state.size() * sizeof(int));
    if (status != cudaSuccess)
    {
        DEBUG(printf("[cuda_play_turn] Failed copy matrix data to constant memory: %d\n", status))
        return false;
    }        

    // First we need to run our cuRAND initialization kernel
    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch();
    auto nano_seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    DEBUG(std::cout << "[cuda_play_turn] Executing 'init_rand_kernel' ... " << std::endl)

    init_rand_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, totalThreadsToLaunch, nano_seed);    
    cudaDeviceSynchronize();

    DEBUG(std::cout << "[cuda_play_turn] Executing 'game_prediction_kernel' ... " << std::endl)

    // Now run the game-prediction engine
    auto shared_memory_size = threadsPerBlock * CONNECT4_MATRIX_SIZE * sizeof(int);
    game_prediction_kernel<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(d_states, d_success_table, our_disc_type);
    cudaDeviceSynchronize();

    DEBUG(std::cout << "[cuda_play_turn] Copying data back to local memory " << std::endl)

    // We now need to copy the data from d_success_table to out_success_per_column
    int* thread_success_table = new int[totalThreadsToLaunch];
    cudaMemcpy(thread_success_table, d_success_table, sizeof(int) * totalThreadsToLaunch, cudaMemcpyDeviceToHost);

    DEBUG(std::cout << "[cuda_play_turn] Data copied back to load memory! " << std::endl)

    // Gather success data
    int highest_score = -1;
    int highest_score_column_index = -1;    

    for (auto i = 0; i < totalThreadsToLaunch; i++)
    {
        // Is this the winning move?
        if (highest_score == -1)
        {
            highest_score = thread_success_table[i];
            highest_score_column_index = i % COL_COUNT;
        }
        else if ((thread_success_table[i] > 0) && thread_success_table[i] < highest_score)
        {
            highest_score = thread_success_table[i];
            highest_score_column_index = i % COL_COUNT;
        }
    }
    
    delete[] thread_success_table;
    
    out_best_move_column = highest_score_column_index;
    out_best_move_row = get_free_row_index_for_column(current_board_state.data(), out_best_move_column);
    out_next_move_wins = highest_score == 1; // We have won!
    out_tie_detected = highest_score == 0; // No thread was able to win (hence all scores being 0), it means we have a tie on our hands!
    
    DEBUG(std::cout << "[cuda_play_turn] Detected highest_score: " << highest_score << " @ column " << highest_score_column_index << std::endl)
    DEBUG(std::cout << "[cuda_play_turn] Game data compiled, freeing memory... " << std::endl)
    
    // Free memory and return
    status = free_memory(d_states, d_success_table);
    if (status != cudaSuccess)
    {
        // We have crashed...
        DEBUG(printf("[cuda_play_turn] Failed to free memory, return code: %d\n", status))
        return false;
    }

    DEBUG(printf("[cuda_play_turn] ** Function call concluded\n"))
    return true;
}
