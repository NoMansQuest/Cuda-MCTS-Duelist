#include "chat_session.h"

using boost::asio::ip::tcp;

chat_session::~ChatSession()
{
    return this->close();    
}

comm_result_t chat_session::start()
{
    return this->do_read();
}

comm_result_t chat_session::send()
{
    boost::asio::post(strand_,
        [self = shared_from_this(), msg = message + "\n"]() mutable
        {
            bool was_empty = self->write_queue_.empty();
            self->write_queue_.push(std::move(msg));

            // Only start writing if this is the first message in queue
            if (was_empty)
            {
                self->do_write();
            }
        });

    return comm_result_t::Success;
}

comm_result_t chat_session::disconnect()
{
    if (!this->is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    boost::asio::post(strand_,
        [self = shared_from_this()]()
        {
            if (self->is_connected_)
            {
                self->is_connected_ = false;
                boost::system::error_code ec;
                self->socket_.shutdown(tcp::socket::shutdown_both, ec);
                self->socket_.close(ec);
                std::cout << "[session] Connection closed by server\n";
            }
        });

    return comm_result_t::Success;
}


comm_result_t chat_session::do_read()
{
    if (!this->is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    auto self = shared_from_this();
    socket_.async_read_some(
        boost::asio::buffer(read_buffer_),
        boost::asio::bind_executor(strand_,
            [this, self](boost::system::error_code ec, std::size_t length)
            {
                if (ec)
                {
                    if (ec != boost::asio::error::operation_aborted)
                    {
                        std::cout << "[session] Read error: " << ec.message() << "\n";
                    }
                    is_connected_ = false;
                    return;
                }

                std::string msg(read_buffer_.data(), length);
                std::cout << "[client] " << msg;

                do_read(); // continue reading
            }));
    
    return comm_result_t::Success;
}

comm_result_t chat_session::do_write()
{
    auto self = shared_from_this();

    if (!is_connected_)
        return comm_result_t::NotConnected;

    if (write_queue_.empty())
        return comm_result_t::Success;

    boost::asio::async_write(
        socket_,
        boost::asio::buffer(write_queue_.front()),
        boost::asio::bind_executor(strand_,
            [this, self](boost::system::error_code ec, std::size_t /*length*/)
            {
                if (ec)
                {
                    if (ec != boost::asio::error::operation_aborted)
                    {
                        std::cout << "[session] Write error: " << ec.message() << "\n";
                    }
                    is_connected_ = false;
                    return;
                }

                write_queue_.pop();

                // If more messages, continue writing
                if (!write_queue_.empty())
                {
                    do_write();
                }
            }));

    return comm_result_t::Success;
}

comm_result_t chat_session::close()
{
    if (!is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    boost::system::error_code ec;
    socket_.shutdown(tcp::socket::shutdown_both, ec);
    socket_.close(ec);
    is_connected_ = false;
    return comm_result_t::Success;
}