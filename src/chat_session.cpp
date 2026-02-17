#include "chat_session.h"

using boost::asio::ip::tcp;

comm_result_t chat_session_t::start()
{
    std::cout << "[chat_session_t::start] entering start() for session " << this << "\n";
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
    if (received_cv_.wait_for(lk, 2s, [self = shared_from_this()] { return !self->received_messages_.empty(); }))
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
    std::cout << "[chat_session_t::do_read] entering do_read() for session " << this << "\n";

    if (!this->is_connected_)
    {
        return comm_result_t::NotConnected;
    }

    // Post the starter — capture raw this (safe, because post is fire-and-forget)
    boost::asio::post(strand_, [this]() {
        std::cout << "[posted read starter] inside posted lambda for session " << this << "\n";

        // Schedule the read — use raw this in the bind_executor lambda too (safe)
        socket_.async_read_some(
            boost::asio::buffer(read_buffer_),
            boost::asio::bind_executor(strand_,
                [this](boost::system::error_code err_code, std::size_t length) mutable
                {
                    // NOW it's safe to get shared_ptr — handler is running asynchronously
                    auto self = this;

                    std::cout << "[async_read_some handler] entered for session " << self << "\n";

                    if (err_code)
                    {
                        if (err_code != boost::asio::error::operation_aborted)
                        {
                            std::cout << "[session] Read error: " << err_code.message() << "\n";
                        }
                        self->is_connected_ = false;
                        return;
                    }

                    // From here on: use self-> everywhere
                    self->remainder_.insert(self->remainder_.end(),
                                            self->read_buffer_.begin(),
                                            self->read_buffer_.begin() + length);

                    while (!self->remainder_.empty())
                    {
                        if (self->remainder_.size() < 1) break;
                        uint8_t msg_size = self->remainder_[0];
                        if (self->remainder_.size() < 1 + msg_size) break;

                        std::vector<uint8_t> msg(self->remainder_.begin() + 1,
                                                 self->remainder_.begin() + 1 + msg_size);

                        {
                            std::lock_guard<std::mutex> lk(self->received_mutex_);
                            self->received_messages_.push(std::move(msg));
                        }
                        self->received_cv_.notify_one();

                        self->remainder_.erase(self->remainder_.begin(),
                                               self->remainder_.begin() + 1 + msg_size);
                    }

                    // Chain next read (still safe)
                    self->do_read();
                }));
        
        std::cout << "[posted read starter] async_read_some has been scheduled\n";
    });

    std::cout << "[do_read] first async read initiation posted to strand\n";
    return comm_result_t::Success;
}

comm_result_t chat_session_t::do_write()
{
    if (!is_connected_)
        return comm_result_t::NotConnected;

    if (write_queue_.empty())
        return comm_result_t::Success;

    boost::asio::async_write(
        socket_,
        boost::asio::buffer(write_queue_.front()),
        boost::asio::bind_executor(strand_,
            [self = shared_from_this()](boost::system::error_code err_code, std::size_t /*length*/)
            {
                std::cout << "[chat_session_t::do_write->async_write] entering async_write for session " << self << "\n";
                if (err_code)
                {
                    if (err_code != boost::asio::error::operation_aborted)
                    {
                        std::cout << "[session] Write error: " << err_code.message() << "\n";
                    }
                    self->is_connected_ = false;
                    return;
                }

                self->write_queue_.pop();

                if (!self->write_queue_.empty())
                {
                    self->do_write();
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