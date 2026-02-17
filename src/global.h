#ifndef _GLOBAL_H__
#define _GLOBAL_H__

#include <boost/asio.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <coroutine>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <array>

/// @brief Number of rows in the game-board
constexpr int game_board_rows = 6;

/// @brief Number of columns in the game-board
constexpr int game_board_columns = 7;

/// @brief Total size (elements) in the CONNECT4 game board.
constexpr int game_board_size = game_board_rows * game_board_columns;

/// @brief Clients disc type
constexpr int client_disc_type = 2;

/// @brief Server's disc type
constexpr int server_disc_type = 1;

#define TO_LINEAR(row, col) ((row * game_board_columns) + col)

/// @brief Help text
const std::string help_text = R"help(
    CUDA Monte-Carlo Tree Search (MCTS) Duelist

    Author: Nasser Ghoseiri
    Date: Feb 10th 2026

    Description:
        This program demonstrates the use of CUDA kernels within a C++ project. The goal is to have two instances
        of this app, one running as server and one as client, compete in a turn-based "Connect 4" game while using
        CUDA GPU acceleration to predict their best next move.

        Interprocess communication is achieved via TCP/IP connection (using boost::asio). The client makes the first
        move and communicates it to the server. The server makes its next move and communicates it back to the client.
        This back-and-forth continues until one party wins, concluding the game session.

        On the CUDA kernel side, each instances runs 7,000 threads (one thousand per possible move on the board) to
        try various combinations of moves using Monte-Carlo algorithm. The column that amasses the highest number of
        predicted wins will then be chosen as the next best move.
    
    Build system:
        This executable is built using CMake, supporting both Windows and Linux. 

    Commandline Arguments:
        --server                : Run instance as server, actively listening for an incoming TCP connection on provided port
        --port PORT_NUMBER      : Port number. In server mode, this is the port we listen to. In client mode, this is the port we connect to.
        --client                : Run instance as client, make a TCP connection to the provided port and IP address.
        --ip IP_ADDRESS         : IP address to connect to (only valid in client mode).

    )help";

#endif