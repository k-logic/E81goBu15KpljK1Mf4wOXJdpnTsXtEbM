#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <random>

namespace chunker {
// CHW形式のエンコード出力を「ピクセルN個単位」でチャンク化
template <typename T = float>
std::vector<std::vector<T>> chunk_by_pixels(
    const std::vector<T>& chw_data,
    int c, int h, int w,
    int pixels_per_chunk
) {
    std::vector<std::vector<T>> chunks;
    int hw = h * w;
    int total_pixels = hw;

    for (int pixel = 0; pixel < total_pixels; pixel += pixels_per_chunk) {
        std::vector<T> chunk;
        int end_pixel = std::min(pixel + pixels_per_chunk, total_pixels);

        for (int p = pixel; p < end_pixel; ++p) {
            int y = p / w;
            int x = p % w;
            for (int ch = 0; ch < c; ++ch) {
                int index = ch * hw + y * w + x;
                chunk.push_back(chw_data[index]);
            }
        }

        chunks.push_back(std::move(chunk));
    }

    return chunks;
}

// チャンク化データを元のCHW形式「1次元ベクトル」に再構築化
template <typename T = float>
std::vector<T> reconstruct_from_chunks(
    const std::vector<std::vector<T>>& chunks,
    int c, int h, int w,
    int pixels_per_chunk = 16
) {
    int hw = h * w;
    std::vector<T> chw_data(c * hw, static_cast<T>(0));

    int pixel_index = 0;

    for (const auto& chunk : chunks) {
        int num_pixels = chunk.size() / c;
        for (int p = 0; p < num_pixels; ++p) {
            if (pixel_index >= hw) break;  // 安全対策
            int y = pixel_index / w;
            int x = pixel_index % w;
            for (int ch = 0; ch < c; ++ch) {
                int dst_index = ch * hw + y * w + x;
                chw_data[dst_index] = chunk[p * c + ch];
            }
            ++pixel_index;
        }
    }

    return chw_data;
}

// チャンクを欠損させ、復元時に欠損箇所は黒（0埋め）にするための情報付き出力
template <typename T = float>
inline std::vector<std::vector<T>> randomly_drop_chunks_with_black_fill(
    const std::vector<std::vector<T>>& chunks,
    float drop_rate = 0.1f
) {
    std::vector<std::vector<T>> result(chunks.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int kept = 0, dropped = 0;

    for (size_t i = 0; i < chunks.size(); ++i) {
        if (dis(gen) > drop_rate) {
            result[i] = chunks[i];
            ++kept;
        } else {
            result[i] = std::vector<T>(chunks[i].size(), static_cast<T>(0));  // 黒で埋める
            ++dropped;
        }
    }

    std::cout << "欠損テスト: " << chunks.size() << " → 残存: " << kept << " / 欠損: " << dropped
              << " (drop_rate=" << drop_rate * 100 << "%)\n";

    return result;
}
}
