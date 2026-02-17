#include "game_helpers.h"

void print_game_board(std::array<int, game_board_size> game_board)
{
    for (auto row = 0; row < game_board_rows; row++)
    {
        std::cout << "    "; // Add a bit of a left-padding
        for (auto col = 0; col < game_board_columns; col++)
        {
            auto slot_value = game_board[TO_LINEAR(row, col)];
            std::cout << (slot_value == client_disc_type ? 'O' : (slot_value == server_disc_type ? 'X' : '-')) << " ";
        }
        std::cout << std::endl;
    }    
    std::cout << std::endl;
}