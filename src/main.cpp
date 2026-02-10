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

#ifndef __cpp_impl_coroutine
#error "Your compiler does not support C++20 coroutines!"
#endif  

using asio::ip::tcp;


void duelist_client_mode(
    boost::asio::io_context& io_context,
    std::string ip_address,
    uint16_t port)
{
    
    boost::system::error_code conn_error_code;
    bool connected = false;

    steady_timer asio::timer(io);

    // start async connect
    boost::asio::async_connect(sock, endpoints,
        [&](const boost::system::error_code& error_code_, auto){
            conn_error_code = error_code_;
            connected = !error_code_;

            // stop the timer if connect finished
            timer.cancel();
        });

    // start timeout timer
    timer.expires_after(timeout);
    timer.async_wait([&](const boost::system::error_code& ec){
        if (!ec && !connected) {

            // Timer fired -> abort the connect
            // closing the socket will cause the connect handler to complete with an error
            sock.close();
        }
    });

    // run until the above handlers finish (this runs handlers on current thread)
    io.run();
    io.restart(); // if io_context will be reused

    tcp::resolver resolver(io_context);
    auto endpoints = resolver.resolve(host, port);
    boost::asio::connect(socket_, endpoints);
    std::cout << "Connected to " << host << ":" << port << "\n\n";

    auto session = std::make_shared<ChatSession>(std::move(socket));
    session->start();
}


void duelist_server_mode(
    boost::asio::io_context& io_context,
    uint16_t port)
{
    try
    {
        boost::asio::io_context io_context;

        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 4444));
        std::cout << "Server listening on port 4444...\n";

        tcp::socket socket(io_context);
        acceptor.accept(socket);

        auto session = std::make_shared<ChatSession>(std::move(socket));
        session->start();

        std::cout << "Client connected. Type messages to send. Type 'disconnect' or 'quit' to close.\n\n";

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

const std::string help_text = R"help(
    CUDA Monte-Carlo Tree Search (MCTS) Duelist\n
    Author: Nasser Ghoseiri
    Date: Feb 10th 2026

    Description:
        This program demonstrates the use of CUDA kernels within a C++ project. The goal is to have
        two instances of this app, one running as server and one as client, play the turn-based "Connect 4" game
        and using CUDA GPU acceleration to predict the best next move.
    
    Build system:
        This executable is built using CMake, supporting both Windows and Linux. 

    Commandline Arguments:
        --server                : Run instance as server, actively listening for an incoming TCP connection on provided port
        --port PORT_NUMBER      : Port number. In server mode, this is the port we listen to. In client mode, this is the port we connect to.
        --client                : Run instance as client, make a TCP connection to the provided port and IP address.
        --ip IP_ADDRESS         : IP address to connect to (only valid in client mode).

    )help";


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
    uint16_t port = 0;
    bool asking_for_help = (argc == 0);
    std::string remote_port;

    if (!parse_arguments(argc, args, remote_ip, port, asking_for_help, server_mode))
    {        
        return 0;
    }

    


    

    return 0;
}