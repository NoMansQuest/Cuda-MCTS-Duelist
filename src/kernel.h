#ifndef CUDA_MCTS_DUELIST_KERNEL_H__
#define CUDA_MCTS_DUELIST_KERNEL_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

/// @brief Kernel initializes cuRAND states for the threads we need
/// @param d_states States for our threads. We need one state per thread we intend to launch.
/// @param total_kernels Total kernels to generate state for (also indicates the size of the @see{states} array.
/// @param seed Randomizer see
__global__ void init_rand_kernel(curandState* d_states, int total_kernels, unsigned long seed);

/// @brief Returns the free row index for a given column.
/// @param d_matrix The 6x7 matrix containing the state
/// @param column Index of the column to check for
/// @return -1 if no free space is available, free-space row index otherwise
__device__ int get_free_row_index_for_column(int* d_matrix, int column);

/// @brief Checks if the just-inserted disc results in a win
/// @note We can determine our own disc type by checking the matrix at new_disc location.
/// @param d_matrix Memory location holding the Connect-4 matrix data (42 entries long, 168 bytes)
/// @param new_disc_row Row where the new disc was placed
/// @param new_disc_column Column where the new disc was placed
/// @return 'True' if the new disc makes us the winner, false otherwise.
__device__ bool check_if_won(int* d_matrix, int new_disc_row, int new_disc_column);

/// @brief Allocate device memory needed by the kernels
/// @param d_states cuRAND states to be allocated (one per thread)
/// @param d_success_table An array of seven entries to hold number of successful wins per move per column
/// @param total_threads Total number of threads we intend to run (which determines the size of the 'd_states')
__host__ cudaError_t allocate_memory(curandState** d_states, int** d_success_table, int totalThreads);

/// @brief Free memory allocated on the device
/// @param d_states cuRAND states
/// @param d_success_table Success table
__host__ cudaError_t free_memory(curandState* d_states, int* d_success_table);

/// @brief Kernel predicting next moves for each of the available columns
/// @note Each kernel determines the column it needs to play first based on its ID modulus 7.
/// @param d_states cuRAND states (one per thread)
/// @param d_success_table An array of 7 integers representing number of wins detected if the next disc was to be inserted in that column. 
/// @param our_disc_type Our disc type (either 1 or 2).
__global__ void game_prediction_kernel(
    curandState* d_states,
    int* d_success_table,
    int our_disc_type);

/// @brief Plays the game based on current board state and disc type
/// @param our_disc_type Our disc type (either 1 or 2).
/// @param out_success_per_column Output: chance of winning per column (7 entries, one per column). The higher the score, the better the move.
/// @param out_best_move_column Output: best column as next move. If -1, no valid moves left (hence a tie).
/// @param out_next_move_wins Output: If true, we have won the match.
/// @return True if procedure runs error free, false indicates a crash/error.
__host__ bool play_game(
    std::vector<std::vector<int>> current_board_state,
    int our_disc_type,
    std::vector<int>& out_success_per_column,
    int& out_best_move_column,
    bool& out_next_move_wins);

#endif