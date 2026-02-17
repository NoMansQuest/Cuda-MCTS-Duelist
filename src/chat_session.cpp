#include "chat_session.h"

using boost::asio::ip::tcp;

comm_result_t chat_session_t::start()
{
    std::cout << "[chat_session_t::start] entering start() for session " << this << "\n";
    return this->do_read();
}

comm_result_t chat_session_t::send_message(const std::vector<uint8_t>& message)
{
    std::cout << "[chat_session_t::send_message] entering for session " << this << "\n";

    if (message.size() > 255)
    {
        throw std::out_of_range("'message' vector could contain maximum 255 bytes.");
    }

    auto self__ = this->self_;
    boost::asio::post(strand_,
        [self__ , message]() mutable
        {
            std::cout << "[chat_session_t::send_message::post] entering for session " << self__ << ", data length:" << message.size() << std::endl;
            std::vector<uint8_t> wrapper(message);
            wrapper.insert(wrapper.begin(), (uint8_t)message.size());

            bool was_empty = self__->write_queue_.empty();
            self__->write_queue_.push(std::move(wrapper));

            if (was_empty) {
                std::cout << "[chat_session_t::send_message::post] calling self->do_write" << std::endl;
                self__->do_write();
            }
        });

    return comm_result_t::Success;
}

comm_result_t chat_session_t::wait_for_message(std::vector<uint8_t>& out_message)
{
    std::cout << "[chat_session_t::wait_for_message] entering for session " << this << std::endl;

    if (!is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    std::cout << "[chat_session_t::wait_for_message] committing to wait..." << this << std::endl;

    std::unique_lock<std::mutex> lk(received_mutex_);
    auto self__ = this->self_;
    if (received_cv_.wait_for(lk, 2s, [self__] { return !self__->received_messages_.empty(); }))
    {
        std::cout << "[chat_session_t::wait_for_message] received data, length: " << received_messages_.size() << std::endl;
        out_message = std::move(received_messages_.front());
        received_messages_.pop();
        return comm_result_t::Success;
    }
    else
    {
        std::cout << "[chat_session_t::wait_for_message] timeout encountered " << std::endl;
        return comm_result_t::Timeout;
    }
}

comm_result_t chat_session_t::disconnect()
{
    std::cout << "[chat_session_t::disconnect] entering for session " << this << std::endl;

    if (!this->is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    std::cout << "[chat_session_t::disconnect] posting operation... " << this << std::endl;

    auto self__ = this->self_;
    boost::asio::post(strand_,
        [self__]()
        {
            std::cout << "[chat_session_t::disconnect::post] entering for session " << self__ << std::endl;
            if (self__->is_connected_)
            {
                std::cout << "[chat_session_t::disconnect::post] initiating disconnect action " << std::endl;
                self__->is_connected_ = false;
                boost::system::error_code err_code;
                self__->socket_.shutdown(tcp::socket::shutdown_both, err_code);
                self__->socket_.close(err_code);
                std::cout << "[session] Connection closed\n";
            }
        });

    return comm_result_t::Success;
}

comm_result_t chat_session_t::do_read()
{
    std::cout << "[chat_session_t::do_read] entering do_read() for session " << this << "\n";

    if (!this->is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    // Schedule the read — use raw this in the bind_executor lambda too (safe)
    auto self__ = this->self_;
    socket_.async_read_some(
        boost::asio::buffer(read_buffer_),
        boost::asio::bind_executor(strand_,
            [self__](boost::system::error_code err_code, std::size_t length) mutable
            {
                // NOW it's safe to get shared_ptr — handler is running asynchronously
                std::cout << "[async_read_some handler] entered for session " << self__ << "\n";

                if (err_code)
                {
                    if (err_code != boost::asio::error::operation_aborted)
                    {
                        std::cout << "[session] Read error: " << err_code.message() << "\n";
                    }
                    self__->is_connected_ = false;
                    return;
                }

                // From here on: use self-> everywhere
                self__->remainder_.insert(self__->remainder_.end(), self__->read_buffer_.begin(), self__->read_buffer_.begin() + length);

                while (!self__->remainder_.empty())
                {
                    if (self__->remainder_.size() < 1) 
                        break;

                    auto msg_size = self__->remainder_[0];

                    if (self__->remainder_.size() < 1 + msg_size) 
                        break;

                    std::vector<uint8_t> msg(
                        self__->remainder_.begin() + 1, 
                        self__->remainder_.begin() + 1 + msg_size);

                    // Protected scope for 'received_message_' access
                    {
                        std::lock_guard<std::mutex> lk(self__->received_mutex_);
                        self__->received_messages_.push(std::move(msg));
                    }

                    self__->received_cv_.notify_one();
                    self__->remainder_.erase(
                        self__->remainder_.begin(),
                        self__->remainder_.begin() + 1 + msg_size);
                }

                // Chain next read (still safe)
                self__->do_read();
            }));
    


    std::cout << "[do_read] first async read initiation posted to strand\n";
    return comm_result_t::Success;
}

comm_result_t chat_session_t::do_write()
{
    std::cout << "[chat_session_t::do_write] entering for session " << this << std::endl;

    if (!is_connected_)
        return comm_result_t::NotConnected;

    if (write_queue_.empty())
        return comm_result_t::Success;

    std::cout << "[chat_session_t::do_write] executing async_write " << std::endl;

    auto self__ = this->self_;        
    boost::asio::async_write(
        socket_,
        boost::asio::buffer(write_queue_.front()),
        boost::asio::bind_executor(strand_,
            [self__](boost::system::error_code err_code, std::size_t /*length*/)
            {
                std::cout << "[chat_session_t::do_write->async_write] entering async_write for session " << self__ << "\n";
                if (err_code)
                {
                    if (err_code != boost::asio::error::operation_aborted)
                    {
                        std::cout << "[session] Write error: " << err_code.message() << "\n";
                    }
                    self__->is_connected_ = false;
                    return;
                }

                self__->write_queue_.pop();

                if (!self__->write_queue_.empty())
                {
                    std::cout << "[chat_session_t::do_write::async_write] repeating do_write " << std::endl;
                    self__->do_write();
                }
            }));

    return comm_result_t::Success;
}

comm_result_t chat_session_t::close()
{
    std::cout << "[chat_session_t::close] entering for session " << this << std::endl;

    if (!is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    std::cout << "[chat_session_t::close] initiating shutdown sequence " << this << std::endl;

    boost::system::error_code ec;
    socket_.shutdown(tcp::socket::shutdown_both, ec);
    socket_.close(ec);
    is_connected_ = false;
    return comm_result_t::Success;
}