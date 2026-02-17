#include "global.h"
#include "kernel.h"
#include "chat_session.h"
#include "session_message.h"

#include "client_mode.h"
#include "server_mode.h"

#ifndef __cpp_impl_coroutine
#error "Your compiler does not support C++20 coroutines!"
#endif  

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
    uint16_t& port,
    bool& asking_for_help,
    bool& server_mode)
{
    bool server_switch = false;
    bool client_switch = false;
    asking_for_help = false;
    remote_ip = "";
    port = 0;

    std::string port_number_in_string = "";

    for (auto arg_index = 0; arg_index < argc; arg_index++) {
        std::string arg(args[arg_index]);

        if ((arg.compare("--help") == 0) || (arg.compare("-H") == 0)) {
            asking_for_help = true;
            break;
        }        

        if (arg.compare("--ip") == 0) {
            if (arg_index == argc - 1){
                std::cout << "Error, a valid IP address must be provided after the '--ip' argument." << std::endl;
                return 0;
            }
            remote_ip = std::string(args[++arg_index]);
            continue;
        }

        if (arg.compare("--port") == 0) {
            if (arg_index == argc - 1){
                std::cout << "Error, a valid port number must be provided after the '--port' argument." << std::endl;
                return 0;
            }            
            port_number_in_string = std::string(args[++arg_index]);            
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
        std::cout << help_text << std::endl;        
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

    if (port_number_in_string == "") {
        std::cout << "Error, port must be defined." << std::endl;
        return false;
    }

    try 
    {
        auto cast_value = std::stoi(port_number_in_string); // may throw
        if (cast_value > 65535 || cast_value < 1) {
            std::cout << "Error, port number out of range (choose between 1 and 65,535)." << std::endl;
            return false;
        }
        port = (uint16_t)cast_value;
    } catch (const std::invalid_argument&) {
        std::cout << "Error, '" << port_number_in_string << "' is not a valid port number" << std::endl;
        return false;
    } catch (const std::out_of_range&) {
        std::cout << "Error, '" << port_number_in_string << "' is out-of-range (min. 0 and max. 65535 is allowed)" << std::endl;
        return false;
    }

    server_mode = server_switch;
    return true;
}

int main(int argc, char *args[])
{
    auto server_mode = false;
    bool asking_for_help = false;
    std::string remote_ip;
    uint16_t port = 0;
    
    if (!parse_arguments(argc, args, remote_ip, port, asking_for_help, server_mode))
    {        
        return 0;
    }

    boost::asio::io_context io_context_;    

    if (server_mode)
        duelist_server_mode(io_context_, port);
    else
        duelist_client_mode(io_context_, std::move(remote_ip), port);

    std::cout << "Application execution concluded." << std::endl;
    return 0;
}