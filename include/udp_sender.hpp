#pragma once

#define ASIO_STANDALONE
#include <config.hpp>
#include <asio/asio.hpp>
#include <asio/awaitable.hpp>
#include <asio/use_awaitable.hpp>
#include <asio/experimental/awaitable_operators.hpp>
#include <optional>
#include <string>
#include <vector>
#include <iostream>
#include <sys/socket.h>

using asio::ip::udp;
using asio::awaitable;
using asio::use_awaitable;
namespace this_coro = asio::this_coro;

class UdpSender {
public:
    // 同期初期化
    /*
    void init_sync(asio::io_context& io, const std::string& ip, uint16_t port) {
        socket_.emplace(io);
        endpoint_ = udp::endpoint(asio::ip::make_address(ip), port);
        socket_->open(udp::v4());

        configure_socket();
    }
    */

    void init_sync(asio::io_context& io, const std::string& host, uint16_t port) {
        socket_.emplace(io);

        udp::resolver resolver(io);
        auto results = resolver.resolve(udp::v4(), host, std::to_string(port));
        endpoint_ = *results.begin();

        socket_->open(udp::v4());

        configure_socket();
    }

    // 同期送信
    void send_sync(const std::vector<uint8_t>& data) {
        socket_->send_to(asio::buffer(data), endpoint_);
    }

    // 非同期初期化
    awaitable<void> init_async(const std::string& ip, uint16_t port) {
        auto executor = co_await this_coro::executor;
        socket_.emplace(executor);
        endpoint_ = udp::endpoint(asio::ip::make_address(ip), port);
        socket_->open(udp::v4());

        configure_socket();
        co_return;
    }

    // 非同期送信
    awaitable<void> send_async(const std::vector<uint8_t>& data) {
        co_await socket_->async_send_to(asio::buffer(data), endpoint_, use_awaitable);
    }

    // クローズ処理
    void close() {
        if (socket_ && socket_->is_open()) socket_->close();
    }

private:
    std::optional<udp::socket> socket_;
    udp::endpoint endpoint_;

    void configure_socket() {
        int fd = socket_->native_handle();

        // 超低遅延向け：バッファを小さく
        int sndbuf = config::UDP_SO_SNDBUF;
        if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)) < 0) {
            perror("setsockopt(SO_SNDBUF) failed");
        } else {
            std::cerr << fmt::format("[INFO] SO_SNDBUF set to {} bytes\n", sndbuf);
        }
    }
};
