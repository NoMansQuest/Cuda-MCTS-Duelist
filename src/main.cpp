#include <asio.hpp>
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/awaitable.hpp>
#include <asio/use_awaitable.hpp>
#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <coroutine>
#include <iostream>
#include <string>
#include <stdexcept>

#include "kernel.h"
#include "chat_session.h"

#ifndef __cpp_impl_coroutine
#error "Your compiler does not support C++20 coroutines!"
#endif  

const std::string help_text = R"help(
    CUDA Monte-Carlo Tree Search (MCTS) Duelist\n
    Author: Nasser Ghoseiri
    Date: Feb 10th 2026

    Description:
        This program demonstrates the use of CUDA kernels within a C++ project. The goal is to have
        two instances of this app, one running as server and one as client, compete in a turn-based "Connect 4" game
        while using CUDA GPU acceleration to predict their best next move.

        Interprocess communication is achieved via TCP/IP connection (using boost::asio). The client makes the first
        move and communicates it to the server. The server makes its next move and communicates it back to the client.
        This back-and-forth continues until one party wins, concluding the game session.

        On the CUDA kernel side, each instances runs 7,000 threads (one thousand per possible move on the board) to try various
        combinations of moves using Monte-Carlo algorithm. The column that amasses the highest number of predicted wins will then
        be chosen as the next best move.
    
    Build system:
        This executable is built using CMake, supporting both Windows and Linux. 

    Commandline Arguments:
        --server                : Run instance as server, actively listening for an incoming TCP connection on provided port
        --port PORT_NUMBER      : Port number. In server mode, this is the port we listen to. In client mode, this is the port we connect to.
        --client                : Run instance as client, make a TCP connection to the provided port and IP address.
        --ip IP_ADDRESS         : IP address to connect to (only valid in client mode).

    )help";


/// @brief Application runs as client.
/// @param io_context Boost::Asio IO context.
/// @param ip_address IP address to connect to.
/// @param port Remote server port to connect to.
void duelist_client_mode(
    boost::asio::io_context& io_context,
    std::string&& ip_address,
    uint16_t port)
{    
    // Introduction
    std::cout << "CUDA MCTS Duelist" << std::endl;
    std::cout << "Starting as client, connecting to " << ip_address << ":" << port << "..." << std::endl;    
    
    try {
        boost::asio::ip::tcp::resolver resolver(io_context);
        boost::system::error_code ec;

        // Resolve the address/port
        auto endpoints = resolver.resolve(ip_address, std::to_string(port), ec);
        if (ec) {
            std::cout << "Resolve failed: " << ec.message() << "\n";
            return;
        }

        // Create socket and connect (blocking)
        boost::asio::ip::tcp::socket socket(io_context);
        boost::asio::connect(socket, endpoints, ec);
        if (ec) {
            std::cout << "Connect failed: " << ec.message() << "\n";
            return;
        }

        // Connected: report remote endpoint
        auto remote_ep = socket.remote_endpoint(ec);
        if (!ec) {
            std::cout << "Connected to " << remote_ep.address().to_string()
                      << ":" << remote_ep.port() << "\n";
        } else {
            std::cout << "Connected (remote endpoint unavailable): " << ec.message() << "\n";
        }

        // TODO: hand `socket` to your ChatSession or other logic here.
        // e.g. auto session = std::make_shared<ChatSession>(std::move(socket));
        // session->start();

        chat_session session(std::move(socket), nullptr);
        session.start();    
        
        // As client, we need to wait for the first move by the remote. Server makes the first move...
        session.



    } catch (const std::exception& ex) {
        std::cerr << "Exception in client: " << ex.what() << "\n";
    }

    auto session = std::make_shared<ChatSession>(std::move(socket));
    session->start();
}


/// @brief Application runs as server
/// @param io_context Boost::Asio IO context.
/// @param port Port to listen to for incoming connection.
void duelist_server_mode(
    boost::asio::io_context& io_context,
    uint16_t port)
{
    // Introduction
    std::cout << "CUDA MCTS Duelist" << std::endl;
    std::cout << "Starting as server, awaiting connection on port " << port << "..." << std::endl;

    try
    {
        boost::asio::io_context io_context;

        boost::asio::ip::tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), port));
        std::cout << "Server listening on port " << port << "..." << std::endl;

        boost::asio::ip::tcp::socket socket(io_context);
        acceptor.accept(socket);

        // Let the user know
        boost::system::error_code remote_endpoint_error_code;
        auto remote_endpoint = socket.remote_endpoint(remote_endpoint_error_code);

        if (!remote_endpoint_error_code) {
            std::cout << "Connection established with " 
                << remote_ep.address().to_string() << ":"
                << remote_ep.port() << "..." 
                << std::endl;
        }
        else{
            std::cout << "Connection established (failed to obtain remote endpoint: " 
                << remote_endpoint_error_code.message() 
                << ")" << std::endl; 
        }        

        chat_session session(std::move(socket), nullptr);
        session->start();        

        // The game starts here... we make the first move, communicate it to the
        // opponent and wait for their action.
        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line == "quit" || line == "exit")
            {
                std::cout << "Shutting down server...\n";
                break;
            }

            if (line == "disconnect")
            {
                std::cout << "Disconnecting client...\n";
                session->disconnect();
                continue;
            }

            if (!line.empty() && session->is_connected())
            {
                session->send(line);
                std::cout << "[you] " << line << "\n";
            }
            else if (!session->is_connected())
            {
                std::cout << "[warning] Client already disconnected\n";
            }
        }

        session->disconnect();
        io_context.stop();
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}

/// @brief Parse arguments passed to the command line.
/// @param argc Number of arguments available in "args"
/// @param args Passed command-line arguments
/// @param remote_ip (Output) Remote IP set by the caller (if any)
/// @param port_number (Output) Port number set by the caller
/// @param asking_for_help (Output) If set, the caller is requesting the help-string.
/// @param server_mode (Output) If set, the instance should run as server mode, otherwise, as client.
/// @return 'True' if the app could proceed, 'false' if execution must be halted.
bool parse_arguments(
    int argc,
    char* args[],
    std::string& remote_ip,
    uint16_t port,
    bool& asking_for_help,
    bool& server_mode)
{
    bool server_switch;
    bool client_switch;
    std::string port_number;

    for (auto arg_index = 0; arg_index < argc; arg_index++) {
        std::string arg(args[arg_index]);

        if ((arg.compare("--help") == 0) || (arg.compare("-H")) {
            asking_for_help = true;
            break;
        }        

        if (arg.compare("--ip") == 0) {
            if (arg_index == argc - 1){
                std::out << "Error, a valid IP address must be provided after the '--ip' argument." << std::endl;
                return 0;
            }
            remote_ip = std::string(args[++arg_index]);
            continue;
        }

        if (arg.compare("--port") == 0) {
            if (arg_index == argc - 1){
                std::out << "Error, a valid port number must be provided after the '--port' argument." << std::endl;
                return 0;
            }            
            remote_port = std::string(args[++arg_index]);
            continue;
        }

        if (arg.compare("--server") == 0) {
            server_switch = true;
        }

        if (arg.compare("--client") == 0) {
            client_switch = true;
        }
    }

    if (asking_for_help) {
        std::out << help_text << std::endl;        
        return false;
    }

    if (server_switch && client_switch) {
        std::cout << "Error, '--server' and '--client' are both set. An instance could run either as a server or as a client." << std::endl;
        return false;
    }

    if (!server_switch && !client_switch) {
        std::cout << "Error, role of instance is undefined (either '--server' or '--client' must be set)." << std::endl;
        return false;
    }


    if (client_switch && remote_ip.empty()) {
        std::cout << "Error, in client mode(i.e. '--client' set), the remote IP address must be passed using the '--ip' switch." << std::endl;
        return false;
    }

    try 
    {
        auto cast_value = std::stoi(port_number); // may throw
        if (cast_value > 65535 || cast_value < 0) {
            throw std::out_of_range("Error, port number out of range");
        }
    } catch (const std::invalid_argument&) {
        std::out << "Error, '" << port_number << "' is not a valid port number" << std::endl;
        return false;
    } catch (const std::out_of_range&) {
        std::out << "Error, '" << port_number << "' is out-of-range (min. 0 and max. 65535 is allowed)" << std::endl;
        return false;
    }

    server_mode = server_switch;
    return true;
}

int main(int argc, char *args[])
{
    auto server_mode = false;
    bool asking_for_help = (argc == 0);
    std::string remote_ip;
    uint16_t port = 0;

    if (!parse_arguments(argc, args, remote_ip, port, asking_for_help, server_mode))
    {        
        return 0;
    }

    boost::asio::io_context io_context_;
    if (server_mode)
        duelist_server_mode(io_context_, port)
    else
        duelist_client_mode(io_context_, std::move(remote_ip), port);    

    return 0;
}