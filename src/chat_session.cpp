#include "chat_session.h"

using boost::asio::ip::tcp;

comm_result_t chat_session_t::start()
{
    return this->do_read();
}

comm_result_t chat_session_t::send_message(const std::vector<uint8_t>& message)
{
    if (message.size() > 255)
    {
        throw std::out_of_range("'message' vector could contain maximum 255 bytes.");
    }

    boost::asio::post(strand_,
        [self = shared_from_this(), message]() mutable
        {
            std::vector<uint8_t> wrapper(message);
            wrapper.insert(wrapper.begin(), (uint8_t)message.size());

            bool was_empty = self->write_queue_.empty();
            self->write_queue_.push(std::move(wrapper));

            if (was_empty)
            {
                self->do_write();
            }
        });

    return comm_result_t::Success;
}

comm_result_t chat_session_t::wait_for_message(std::vector<uint8_t>& out_message)
{
    if (!is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    std::unique_lock<std::mutex> lk(received_mutex_);
    if (received_cv_.wait_for(lk, 2s, [this] { return !received_messages_.empty(); }))
    {
        out_message = std::move(received_messages_.front());
        received_messages_.pop();
        return comm_result_t::Success;
    }
    else
    {
        return comm_result_t::Timeout;
    }
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
                std::cout << "[session] Connection closed\n";
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

                // Append new data to remainder
                remainder_.insert(remainder_.end(), read_buffer_.begin(), read_buffer_.begin() + length);

                // Parse complete messages
                while (!remainder_.empty())
                {
                    if (remainder_.size() < 1) break;  // Need at least size byte
                    uint8_t msg_size = remainder_[0];
                    if (remainder_.size() < 1 + msg_size) break;  // Not enough for full message

                    std::vector<uint8_t> msg(remainder_.begin() + 1, remainder_.begin() + 1 + msg_size);
                    {
                        std::lock_guard<std::mutex> lk(received_mutex_);
                        received_messages_.push(std::move(msg));
                    }
                    received_cv_.notify_one();

                    // Erase processed data
                    remainder_.erase(remainder_.begin(), remainder_.begin() + 1 + msg_size);
                }

                // Continue reading
                do_read();
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