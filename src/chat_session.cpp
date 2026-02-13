#include "chat_session.h"

using boost::asio::ip::tcp;


comm_result_t chat_session_t::start()
{
    return this->do_read();
}

comm_result_t chat_session_t::send_message(const std::vector<uint8_t>& message)
{
    boost::asio::post(strand_,
        [self = shared_from_this(), message]() mutable
        {
            std::vector<uint8_t> wrapper(message);
            wrapper.insert(wrapper.begin(), message.size());           

            bool was_empty = self->write_queue_.empty();
            self->write_queue_.push(std::move(wrapper));

            // Only start writing if this is the first message in queue
            if (was_empty)
            {
                self->do_write();
            }
        });

    return comm_result_t::Success;
}

comm_result_t chat_session_t::wait_for_message(std::vector<uint8_t>& out_message)
{

}

comm_result_t chat_session_t::disconnect()
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
                boost::system::error_code err_code;
                self->socket_.shutdown(tcp::socket::shutdown_both, err_code);
                self->socket_.close(err_code);
                std::cout << "[session] Connection closed by server\n";
            }
        });

    return comm_result_t::Success;
}


comm_result_t chat_session_t::do_read()
{
    if (!this->is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    auto self = shared_from_this();
    socket_.async_read_some(
        boost::asio::buffer(read_buffer_),
        boost::asio::bind_executor(strand_,
            [this, self](boost::system::error_code err_code, std::size_t length)
            {
                if (err_code)
                {
                    if (err_code != boost::asio::error::operation_aborted)
                    {
                        std::cout << "[session] Read error: " << err_code.message() << "\n";
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

comm_result_t chat_session_t::do_write()
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
            [this, self](boost::system::error_code err_code, std::size_t /*length*/)
            {
                if (err_code)
                {
                    if (err_code != boost::asio::error::operation_aborted)
                    {
                        std::cout << "[session] Write error: " << err_code.message() << "\n";
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

comm_result_t chat_session_t::close()
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