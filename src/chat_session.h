#ifndef _CHAT_SESSION_H__
#define _CHAT_SESSION_H__

#include <asio.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <queue>
#include "comm_result.h"

using boost::asio::ip::tcp;
using namespace std::literals;

/// @brief Entity represents an active chat session
class chat_session_t : public std::enable_shared_from_this<chat_session_t>
{

private:
    tcp::socket socket_;
    boost::asio::strand<boost::asio::io_context::executor_type> strand_;
    std::array<char, 1024> read_buffer_{};
    std::queue<std::vector<uint8_t>> write_queue_;
    std::mutex queue_mutex_; 
    std::atomic<bool> is_connected_;
    void* user_data_ = nullptr;

public:
    /// @brief Default constructor
    /// @param socket TCP socket associated with this session.
    /// @param user_data User data associated with this session (in our case, it'll be the Connect4 matrix).
    explicit chat_session_t(
            tcp::socket socket,
            void* user_data)
        : socket_(std::move(socket)),
          strand_(boost::asio::make_strand(socket_.get_executor())),
          is_connected_(true),
          user_data_(user_data)
    {}

    /// @brief Default destructor
    /// @note We do need to close the connection just to be sure.
    chat_session_t::~chat_session_t()
    {
        this->close();
    }

    /// @brief Starts the session (well, the reading part)    
    comm_result_t start();

    /// @brief Send a message (thread-safe) to the remote party.
    comm_result_t send_message(const std::vector<uint8_t>& msg);

    /// @brief Wait for remote party to 
    comm_result_t wait_for_message(std::vector<uint8_t>& out_message);    

    /// @brief Gracefully close the connection. Async read operation is also stopped.
    comm_result_t disconnect();

    /// @brief Returns a value indicating whether the connection is there.    
    [[nodiscard]] bool is_connected() const
    {
        return this->is_connected_;
    }

    /// @brief Returns the user data associated with this chat session
    [[nodiscard]] void* user_data() const
    {
        return this->user_data_;
    }

private:
    /// @brief Read data from TCP socket
    comm_result_t do_read();

    /// @brief Write data in our queue to TCP socket.
    comm_result_t do_write();

    /// @brief Close the connection.
    comm_result_t close();
};

#endif