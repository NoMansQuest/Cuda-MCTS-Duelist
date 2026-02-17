#include "server_mode.h"
#include "chat_session.h"
#include "game_helpers.h"
#include "game_engine.h"

void duelist_server_mode( boost::asio::io_context& io_context, uint16_t port)
{
    // Introduction
    std::cout << "CUDA MCTS Duelist starting..." << std::endl;
    std::cout << "Starting as server, awaiting connection on port " << port << "..." << std::endl;

    try
    {        
        boost::asio::ip::tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), port));
        std::cout << "Server listening on port " << port << "..." << std::endl;

        boost::asio::ip::tcp::socket socket(io_context);
        acceptor.accept(socket);

        // Let the user know
        boost::system::error_code remote_endpoint_error_code;
        auto remote_endpoint = socket.remote_endpoint(remote_endpoint_error_code);

        if (!remote_endpoint_error_code) {
            std::cout << "Connection established with " 
                << remote_endpoint.address().to_string() << ":"
                << remote_endpoint.port() << "..." 
                << std::endl;
        }
        else{
            std::cout << "Connection established (failed to obtain remote endpoint: " 
                << remote_endpoint_error_code.message() 
                << ")" << std::endl; 
        }        

        auto session = std::make_shared<chat_session_t>(std::move(socket), nullptr);
        session->updated_self(session);
        session->start();
        
        // As client, we need to wait for the first move by the remote. Server makes the first move...
        std::array<int, game_board_size> game_board_linear{};
        run_game(session, game_board_linear, true);
        
        // The game is over, advise the console and print the final state of the game board
        std::cout << "Final state of the game-board: " << std::endl << std::endl;
        print_game_board(game_board_linear);

    } catch (const std::exception& ex) {
        std::cerr << "Exception in client: " << ex.what() << "\n";
    }
}
