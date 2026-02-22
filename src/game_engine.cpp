
#include "game_engine.h"
#include "comm_result.h"
#include "session_message.h"
#include "kernel.h"


int get_free_row_on_column(std::array<int, game_board_size>& game_board_linear, int column)
{
    for (auto row = game_board_rows - 1; row >= 0; row--)
    {
        if (game_board_linear[(row * game_board_columns) + column] == 0)        
        {
            return row;
        }
    }

    return -1;
}

void run_game(
    std::shared_ptr<chat_session_t> session,
    std::array<int, game_board_size>& game_board_linear,
    bool is_server)
{    
    std::vector<uint8_t> incoming_message_raw;
    auto game_ended = false; 
    auto our_disc_type = is_server ? server_disc_type : client_disc_type;
    auto opponent_disc_type = is_server ? client_disc_type : server_disc_type;
    auto incoming_message_result = comm_result_t::Success;

    // If we're the server, we must make the first move
    if (is_server)
    {
        int chosen_column = rand() % game_board_columns;
        int chosen_row = game_board_rows - 1; // The bottom-most row
        game_board_linear[TO_LINEAR(chosen_row, chosen_column)] = our_disc_type;
        session_message_t msg { chosen_row, chosen_column, game_state_t::Playing };
        session->send_message(msg);
    }

    // Game loop
    while(incoming_message_result == comm_result_t::Success && !game_ended)
    {
        incoming_message_result = session->wait_for_message(incoming_message_raw);

        // Deserialize the incoming message
        auto incoming_message = session_message_t::Deserialize(incoming_message_raw);
        std::cout << "Opponent played row " << incoming_message.played_row << ", column " << incoming_message.played_column << std::endl;

        auto free_row_on_column = get_free_row_on_column(game_board_linear, incoming_message.played_column);
        if (free_row_on_column > incoming_message.played_row)
        {
            std::cout << "ERROR! Opponent played row " << incoming_message.played_row << " on column " << incoming_message.played_column << " but row " << free_row_on_column << " was free." << std::endl;
        }

        // We update our game board to include opponent's move
        game_board_linear[TO_LINEAR(incoming_message.played_row, incoming_message.played_column)] = opponent_disc_type;

        if (incoming_message.game_state == game_state_t::PlayerHasWon) {
            std::cout << "Opponent has won!" << std::endl;
            break;
        }

        if (incoming_message.game_state == game_state_t::TieDetected) {
            std::cout << "Tie detected!" << std::endl;
            break;
        }

        // Note: The client discs are marked as 2, server discs are marked as 1.            
        int best_move_column = 0;
        int best_move_row = 0;
        auto next_move_wins = false;
        auto tie_detected = false;
        
        auto play_game_result = cuda_play_turn(
            game_board_linear,
            our_disc_type,
            best_move_row,
            best_move_column,
            next_move_wins,
            tie_detected);

        if (!play_game_result) {
            std::cout << "CUDA operation failed, aborting!" << std::endl;
            return;
        }

        free_row_on_column = get_free_row_on_column(game_board_linear, best_move_column);
        if (free_row_on_column > best_move_row)
        {
            std::cout << "ERROR! We played row " << best_move_row << " on column " << best_move_column << " but row " << free_row_on_column << " was free." << std::endl;
        }

        // We need to send a message to the remote player
        auto game_state = next_move_wins ? game_state_t::PlayerHasWon : (tie_detected ? game_state_t::TieDetected : game_state_t::Playing);
        switch (game_state)
        {
            case game_state_t::Playing:
                game_board_linear[TO_LINEAR(best_move_row, best_move_column)] = our_disc_type;
                std::cout << " Played row " << best_move_row << ", column " << best_move_column << std::endl;
                break;
                
            case game_state_t::PlayerHasWon:
                game_board_linear[TO_LINEAR(best_move_row, best_move_column)] = our_disc_type;
                game_ended = true;
                std::cout << " Played row " << best_move_row << ", column " << best_move_column << " - We have won!" << std::endl;
                break;
                
            default: // Tie
                game_ended = true;
                std::cout << " Tie detected! " << best_move_column << std::endl;
                break;
        }
        
        // Send the message to the remote party
        session_message_t msg { best_move_row, best_move_column, game_state };
        session->send_message(msg);
    }
}