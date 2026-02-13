#ifndef _CLIENT_MODE_H__
#define _CLIENT_MODE_H__

#include "global.h"

/// @brief Application running in client mode
/// @param io_context Boost::Asio IO context
/// @param ip_address IP address of the server
/// @param port Port of the server
void duelist_client_mode(boost::asio::io_context& io_context, std::string&& ip_address, uint16_t port);

#endif