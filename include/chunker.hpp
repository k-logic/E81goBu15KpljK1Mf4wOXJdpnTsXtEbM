#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <random>
#include <span>

namespace chunker {
// CHW形式のエンコード出力を「ピクセルN個単位」でチャンク化
std::vector<std::vector<float>> chunk_by_pixels(
    const std::vector<float>& chw_data,
    int c, int h, int w,
    int pixels_per_chunk
) {
    std::vector<std::vector<float>> chunks;
    chunks.reserve((h * w + pixels_per_chunk - 1) / pixels_per_chunk);

    const int hw = h * w;
    const int total_pixels = hw;

    for (int pixel = 0; pixel < total_pixels; pixel += pixels_per_chunk) {
        int end_pixel = std::min(pixel + pixels_per_chunk, total_pixels);

        std::vector<float> chunk;
        chunk.reserve((end_pixel - pixel) * c); // 再確保防止

        for (int p = pixel; p < end_pixel; ++p) {
            int base_hw = (p / w) * w + (p % w); // y*w + x
            for (int ch = 0; ch < c; ++ch) {
                chunk.push_back(chw_data[ch * hw + base_hw]);
            }
        }

        chunks.push_back(std::move(chunk));
    }

    return chunks;
}

// HWC配列をピクセルN個単位で分割
inline void chunk_by_pixels_hwc(
    const std::vector<float>& hwc,       // 入力 HWC 配列
    int c,                               // チャンネル数
    int h,                               // 高さ
    int w,                               // 幅
    int pixels_per_chunk,                // 1チャンクのピクセル数
    std::vector<std::vector<float>>& chunks // 出力チャンク配列（再利用）
) {
    if (c <= 0 || h <= 0 || w <= 0 || pixels_per_chunk <= 0) {
        throw std::invalid_argument("Invalid argument");
    }

    const int total_pixels = h * w;
    const size_t expected_size = static_cast<size_t>(total_pixels) * c;
    if (hwc.size() != expected_size) {
        throw std::invalid_argument("hwc size mismatch");
    }

    const int num_chunks = (total_pixels + pixels_per_chunk - 1) / pixels_per_chunk;
    chunks.resize(num_chunks);

    const float* base = hwc.data();

    for (int chunk_idx = 0, pixel = 0; pixel < total_pixels; ++chunk_idx, pixel += pixels_per_chunk) {
        const int end_pixel = std::min(pixel + pixels_per_chunk, total_pixels);
        const size_t start_idx = static_cast<size_t>(pixel) * c;
        const size_t end_idx   = static_cast<size_t>(end_pixel) * c;
        const size_t elem_cnt  = end_idx - start_idx;

        auto& chunk = chunks[chunk_idx];
        chunk.resize(elem_cnt);
        std::copy(base + start_idx, base + end_idx, chunk.begin());
    }
}

// ゼロコピー版：チャンク化データを元のCHW形式「1次元ベクトル」に再構築化
template <typename T = float>
void reconstruct_from_chunks(
    const std::vector<std::vector<T>>& chunks,
    T* chw_data,
    int c, int h, int w,
    int pixels_per_chunk = 16
) {
    const int hw = h * w;
    int pixel_index = 0;

    for (const auto& chunk : chunks) {
        const int num_pixels = chunk.size() / c;
        for (int p = 0; p < num_pixels; ++p) {
            if (pixel_index >= hw) break;
            const int y = pixel_index / w;
            const int x = pixel_index % w;
            for (int ch = 0; ch < c; ++ch) {
                const int dst_index = ch * hw + y * w + x;
                chw_data[dst_index] = chunk[p * c + ch];
            }
            ++pixel_index;
        }
    }
}
}
