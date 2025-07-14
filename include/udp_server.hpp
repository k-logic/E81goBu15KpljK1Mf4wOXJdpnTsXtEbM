#pragma once

#define ASIO_STANDALONE
#include <asio/asio.hpp>
#include <array>
#include <atomic>
#include <functional>
#include <iostream>
#include <string>
#include <format>
#include <sys/socket.h>

using asio::ip::udp;
using asio::awaitable;
using asio::use_awaitable;
namespace this_coro = asio::this_coro;

class UdpServer {
public:
    using ReceiveHandler = std::function<void(const udp::endpoint&, const std::vector<char>&)>;

    UdpServer(asio::io_context& io, uint16_t port)
        : executor_(io.get_executor()),
          socket_(executor_, udp::endpoint(udp::v4(), port)),
          running_(true)
    {
        configure_socket(socket_);
    }

    awaitable<void> start(ReceiveHandler handler) {
        try {
            std::cout << std::format("UDP server started on port {}\n", socket_.local_endpoint().port());
            co_await receive_loop(std::move(handler));
        } catch (const std::exception& e) {
            std::cerr << "UDP server error: " << e.what() << "\n";
        }
        co_return;
    }

    void stop() {
        running_ = false;
        if (socket_.is_open()) socket_.close();
    }

private:
    void configure_socket(udp::socket& sock) {
        int rcvbuf_size = 4 * 1024 * 1024;  // 4MB
        int fd = sock.native_handle();
        if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf_size, sizeof(rcvbuf_size)) < 0) {
            perror("setsockopt(SO_RCVBUF) failed");
        } else {
            std::cout << std::format("SO_RCVBUF set to {} bytes\n", rcvbuf_size);
        }
    }

    awaitable<void> receive_loop(ReceiveHandler handler) {
        while (running_) {
            co_await receive_once(handler);
        }
    }

    awaitable<void> receive_once(ReceiveHandler& handler) {
        static thread_local std::array<char, 1500> buffer;
        udp::endpoint sender;

        std::size_t bytes = co_await socket_.async_receive_from(
            asio::buffer(buffer), sender, use_awaitable);

        handler(sender, {buffer.data(), buffer.data() + bytes});
        co_return;
    }

    asio::any_io_executor executor_;
    udp::socket socket_;
    std::atomic<bool> running_;
};
