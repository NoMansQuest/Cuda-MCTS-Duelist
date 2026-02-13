#ifndef _SERVER_MODE_H__
#define _SERVER_MODE_H__

#include "global.h"

/// @brief Application running as server
/// @param io_context Boost::Asio IO context
/// @param port Port of the server
void duelist_server_mode(boost::asio::io_context& io_context, uint16_t port);

#endif