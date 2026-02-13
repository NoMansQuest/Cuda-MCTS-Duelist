#include "client_mode.h"
#include "chat_session.h"
#include "game_helpers.h"
#include "game_engine.h"

/// @brief Application runs as client.
/// @param io_context Boost::Asio IO context.
/// @param ip_address IP address to connect to.
/// @param port Remote server port to connect to.
void duelist_client_mode(boost::asio::io_context& io_context, std::string&& ip_address, uint16_t port)
{    
    // Introduction    
    std::cout << "Starting as client, connecting to " << ip_address << ":" << port << "..." << std::endl;    
    
    try 
    {
        boost::asio::ip::tcp::resolver resolver(io_context);
        boost::system::error_code boost_error_code;

        // Resolve the address/port
        auto endpoints = resolver.resolve(ip_address, std::to_string(port), boost_error_code);
        if (boost_error_code) {
            std::cout << "Resolution failed: " << boost_error_code.message() << "\n";
            return;
        }

        // Create socket and connect (blocking)
        boost::asio::ip::tcp::socket socket(io_context);
        boost::asio::connect(socket, endpoints, boost_error_code);
        if (boost_error_code) {
            std::cout << "Connect failed: " << boost_error_code.message() << "\n";
            return;
        }

        // Connected: report remote endpoint
        auto remote_ep = socket.remote_endpoint(boost_error_code);
        if (!boost_error_code) {
            std::cout << "Connected to " << remote_ep.address().to_string() << ":" << remote_ep.port() << "\n";
        } else {
            std::cout << "Connected (remote endpoint unavailable): " << boost_error_code.message() << "\n";
        }

        chat_session_t session(std::move(socket), nullptr);
        session.start();    
        
        // As client, we need to wait for the first move by the remote. Server makes the first move...
        std::array<int, game_board_size> game_board_linear{};
        run_game(session, game_board_linear, false);
        
        // The game is over, advise the console and print the final state of the game board
        std::cout << "Final state of the game-board: " << std::endl << std::endl;
        print_game_board(game_board_linear);

    } catch (const std::exception& ex) {
        std::cerr << "Exception in client: " << ex.what() << "\n";
    }
}