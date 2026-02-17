
#include "game_engine.h"
#include "comm_result.h"
#include "session_message.h"
#include "kernel.h"

void run_game(
    chat_session_t& session,
    std::array<int, game_board_size> game_board_linear,
    bool is_server)
{
    comm_result_t incoming_message_result;
    std::vector<uint8_t> incoming_message_raw;
    auto game_ended = false; 
    auto disc_type = is_server ? client_disc_type : server_disc_type;

    for(incoming_message_result = session.wait_for_message(incoming_message_raw);
        incoming_message_result == comm_result_t::Success && !game_ended;
        incoming_message_result = session.wait_for_message(incoming_message_raw))
    {
        // Deserialize the incoming message
        auto incoming_message = session_message_t::Deserialize(incoming_message_raw);
        std::cout << "Opponent played row " << incoming_message.played_row << ", column " << incoming_message.played_column << std::endl;

        if (incoming_message.game_state == game_state_t::PlayerHasWon) {
            std::cout << "Opponent has won!" << std::endl;
            break;
        }

        if (incoming_message.game_state == game_state_t::TieDetected) {
            std::cout << "Tie detected!" << std::endl;
            break;
        }

        // We update our game board to include opponent's move
        game_board_linear[TO_LINEAR(incoming_message.played_row, incoming_message.played_column)] = server_disc_type;

        // Note: The client discs are marked as 2, server discs are marked as 1.            
        int best_move_column = 0;
        int best_move_row = 0;
        auto next_move_wins = false;
        auto tie_detected = false;

        auto play_game_result = cuda_play_turn(game_board_linear, disc_type, best_move_row, best_move_column, next_move_wins, tie_detected);

        if (!play_game_result) {
            std::cout << "CUDA operation failed, aborting!" << std::endl;
            return;
        }                        

        // We need to send a message to the remote player
        auto game_state = next_move_wins ? game_state_t::PlayerHasWon : (tie_detected ? game_state_t::TieDetected : game_state_t::Playing);
        switch (game_state)
        {
            case game_state_t::Playing:
                game_board_linear[TO_LINEAR(best_move_row, best_move_column)] = disc_type;
                std::cout << "Played row " << best_move_row << ", column " << best_move_column << std::endl;
                break;
                
            case game_state_t::PlayerHasWon:
                game_board_linear[TO_LINEAR(best_move_row, best_move_column)] = disc_type;
                game_ended = true;
                std::cout << "Played row " << best_move_row << ", column " << best_move_column << " - We have won!" << std::endl;
                break;
                
            default: // Tie
                game_ended = true;
                std::cout << "Tie detected! " << best_move_column << std::endl;
                break;
        }
        
        // Send the message to the remote party
        session_message_t msg { best_move_row, best_move_column, game_state };
        session.send_message(msg);
    }
}