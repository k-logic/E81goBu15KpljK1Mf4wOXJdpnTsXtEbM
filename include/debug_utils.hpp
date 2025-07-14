#pragma once
#include <vector>
#include <iostream>
#include <cstdio>
#include <stdfloat>

namespace debug_utils {

// チャンク情報を出力
template <typename T = float>
inline void print_chunk_info(const std::vector<std::vector<T>>& chunks, int chunk_h, int chunk_w) {
    std::cout << "Total chunks: " << chunks.size() << "\n";
    if (!chunks.empty()) {
        size_t byte_size = chunks[0].size() * sizeof(T);
        std::cout << "Chunk size: " << chunks[0].size() << " floats (First chunk size: " << byte_size << " byte)\n";
    }
}

// 量子化データの16進表示（uint8_t固定なのでテンプレート不要）
inline void print_quantized_hex(const std::vector<uint8_t>& data) {
    std::cout << "Quantized data (hex): [" << data.size() << " bytes]\n";
    for (size_t i = 0; i < data.size(); ++i) {
        std::printf("%02X ", data[i]);
        if ((i + 1) % 16 == 0) std::cout << "\n";
    }
    if (data.size() % 16 != 0) std::cout << "\n";
}

}
