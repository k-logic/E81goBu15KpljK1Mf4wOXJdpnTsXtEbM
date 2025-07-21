#pragma once
#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <vector>
#include <iostream>
#include <cstdio>
#include <stdfloat>

namespace debug_utils {

// チャンク情報を出力
template <typename T = float>
inline void print_chunk_info(const std::vector<std::vector<T>>& chunks, int chunk_h, int chunk_w) {
    std::cout << fmt::format("Total chunks: {}\n", chunks.size());
    if (!chunks.empty()) {
        size_t byte_size = chunks[0].size() * sizeof(T);
        std::cout << fmt::format("Chunk size: {} floats (First chunk size: {} byte)\n", chunks[0].size(), byte_size);

    }
}

// 量子化データの16進表示（uint8_t固定なのでテンプレート不要）
inline void print_quantized_hex(const std::vector<uint8_t>& data) {
    std::cout << fmt::format("Quantized data (hex): [{} bytes]\n", data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        std::printf("%02X ", data[i]);
        if ((i + 1) % 16 == 0) std::cout << "\n";
    }
    if (data.size() % 16 != 0) std::cout << "\n";
}

// ビット列調査
inline void print_float_bit_pattern(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    std::cout << fmt::format("float: {:f} → bits: 0x{:08X}\n", val, bits);
}
}
