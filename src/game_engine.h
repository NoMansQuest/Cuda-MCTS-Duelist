#ifndef _GAME_ENGINE_H__
#define _GAME_ENGINE_H__

#include "global.h"
#include "chat_session.h"

/// @brief Play the game to the end
/// @param session Chat session, used to communicate moves to the remote party
/// @param game_board_linear Game board matrix in an array (2D-matrix flattened)
/// @param make_first_move Make the first move? (this applies to the server)
void run_game(
    chat_session_t& session,
    std::array<int, game_board_size> game_board_linear,
    bool make_first_move);

#endif