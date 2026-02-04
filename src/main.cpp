#include <asio.hpp>
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/awaitable.hpp>
#include <asio/use_awaitable.hpp>
#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <coroutine>
#include <iostream>

#ifndef __cpp_impl_coroutine
#error "Your compiler does not support C++20 coroutines!"
#endif  

using asio::ip::tcp;

// Coroutine to handle an individual client session
asio::awaitable<void> echo(tcp::socket socket) {
    try {
        char data[1024];
        for (;;) {
            // Read data from client
            std::size_t n = co_await socket.async_read_some(asio::buffer(data), asio::use_awaitable);
            // Write it back (Echo)
            co_await async_write(socket, asio::buffer(data, n), asio::use_awaitable);
        }
    } catch (std::exception& e) {
        std::printf("Session closed: %s\n", e.what());
    }
}

// Coroutine to listen for new connections
asio::awaitable<void> listener() {
    auto executor = co_await asio::this_coro::executor;
    tcp::acceptor acceptor(executor, {tcp::v4(), 55555});
    
    for (;;) {
        tcp::socket socket = co_await acceptor.async_accept(asio::use_awaitable);
        // Spawn a new coroutine for each connection
        asio::co_spawn(executor, echo(std::move(socket)), asio::detached);
    }
}

int main() {
    asio::io_context io_context;
    asio::co_spawn(io_context, listener(), asio::detached);    
    io_context.run();
    return 0;
}