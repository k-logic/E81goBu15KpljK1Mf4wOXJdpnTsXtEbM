#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <random>
#include <span>

namespace chunker {
// HWC配列をピクセルN個単位で分割
inline void chunk_by_pixels_hwc(
    const std::vector<uint8_t>& hwc,       // 入力 HWC 配列
    int c,                                 // チャンネル数
    int h,                                 // 高さ
    int w,                                 // 幅
    int pixels_per_chunk,                  // 1チャンクのピクセル数
    std::vector<std::vector<uint8_t>>& chunks // 出力チャンク配列（再利用）
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

    for (int chunk_idx = 0, pixel = 0; pixel < total_pixels; ++chunk_idx, pixel += pixels_per_chunk) {
        const int end_pixel = std::min(pixel + pixels_per_chunk, total_pixels);
        const size_t start_idx = static_cast<size_t>(pixel) * c;
        const size_t end_idx   = static_cast<size_t>(end_pixel) * c;

        auto& chunk = chunks[chunk_idx];
        chunk.assign(hwc.begin() + start_idx, hwc.begin() + end_idx);
    }
}

// chunks(=HWCでピクセル連続) → HWC一次元配列 へ再構築
inline void reconstruct_from_chunks_hwc(
    const std::vector<std::vector<uint8_t>>& chunks,  // ソート済み
    const std::vector<bool>& received_flags,          // チャンク到着フラグ
    uint8_t* hwc_data,                                // 出力バッファ (size = h*w*c)
    int c, int h, int w
) {
    const int hw = h * w;
    int pixel_index = 0;

    // 出力バッファを黒 (0) で初期化
    std::fill(hwc_data, hwc_data + static_cast<size_t>(hw) * c, 0);

    // チャンクごとに復元
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const auto& chunk = chunks[ci];
        const int num_pixels = static_cast<int>(chunk.size()) / c;

        // 欠損チャンクは黒のままスキップ
        if (ci >= received_flags.size() || !received_flags[ci]) {
            pixel_index += num_pixels;
            continue;
        }

        // 正常受信したチャンクのみ書き込み
        for (int p = 0; p < num_pixels; ++p) {
            if (pixel_index >= hw) break;
            for (int ch = 0; ch < c; ++ch) {
                hwc_data[pixel_index * c + ch] = chunk[p * c + ch];
            }
            ++pixel_index;
        }
    }
}

inline void chunk_by_tiles_hwc(
    const std::vector<uint8_t>& hwc,
    int c, int h, int w,
    int tile_w, int tile_h,
    std::vector<std::vector<uint8_t>>& chunks
) {
    if (c <= 0 || h <= 0 || w <= 0 || tile_w <= 0 || tile_h <= 0) {
        throw std::invalid_argument("Invalid argument");
    }

    const size_t expected_size = static_cast<size_t>(h) * w * c;
    if (hwc.size() != expected_size) {
        throw std::invalid_argument("hwc size mismatch");
    }

    const int tiles_x = (w + tile_w - 1) / tile_w;
    const int tiles_y = (h + tile_h - 1) / tile_h;
    chunks.clear();
    chunks.reserve(tiles_x * tiles_y);

    const uint8_t* base = hwc.data();

    for (int ty = 0; ty < tiles_y; ++ty) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            int start_x = tx * tile_w;
            int start_y = ty * tile_h;
            int end_x   = std::min(start_x + tile_w, w);
            int end_y   = std::min(start_y + tile_h, h);

            std::vector<uint8_t> chunk;
            chunk.reserve((end_y - start_y) * (end_x - start_x) * c);

            for (int y = start_y; y < end_y; ++y) {
                const uint8_t* row_ptr = base + (y * w * c);
                for (int x = start_x; x < end_x; ++x) {
                    const uint8_t* pixel_ptr = row_ptr + (x * c);
                    chunk.insert(chunk.end(), pixel_ptr, pixel_ptr + c);
                }
            }
            chunks.push_back(std::move(chunk));
        }
    }
}

inline void reconstruct_from_tiles_hwc(
    const std::vector<std::vector<uint8_t>>& chunks, // タイルごとのデータ
    const std::vector<bool>& received_flags,         // タイル到着フラグ
    uint8_t* hwc_data,                               // 出力HWCバッファ
    int c, int h, int w,                             // チャンネル数, 高さ, 幅
    int tile_w, int tile_h                           // タイル幅・高さ（ピクセル単位）
) {
    const int tiles_x = (w + tile_w - 1) / tile_w;   // 横方向タイル数
    const int tiles_y = (h + tile_h - 1) / tile_h;   // 縦方向タイル数
    const int expected_tile_count = tiles_x * tiles_y;

    if (chunks.size() != static_cast<size_t>(expected_tile_count) ||
        received_flags.size() != static_cast<size_t>(expected_tile_count)) {
        std::cerr << "[WARN] Tile count mismatch: expected "
                  << expected_tile_count
                  << " but got chunks=" << chunks.size()
                  << ", flags=" << received_flags.size()
                  << " — missing tiles will be filled black.\n";
    }

    // 出力バッファを黒 (0) で初期化
    std::fill(hwc_data, hwc_data + static_cast<size_t>(h) * w * c, 0);

    for (int ty = 0; ty < tiles_y; ++ty) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            int tile_index = ty * tiles_x + tx;

            if (tile_index >= static_cast<int>(chunks.size()) ||
                tile_index >= static_cast<int>(received_flags.size()) ||
                !received_flags[tile_index]) {
                // 欠損タイルはスキップ（黒のまま）
                continue;
            }

            const auto& tile = chunks[tile_index];
            int tile_pixels_x = std::min(tile_w, w - tx * tile_w);
            int tile_pixels_y = std::min(tile_h, h - ty * tile_h);

            // タイルを出力バッファにコピー
            for (int py = 0; py < tile_pixels_y; ++py) {
                for (int px = 0; px < tile_pixels_x; ++px) {
                    int global_x = tx * tile_w + px;
                    int global_y = ty * tile_h + py;
                    size_t dst_index = (static_cast<size_t>(global_y) * w + global_x) * c;
                    size_t src_index = (static_cast<size_t>(py) * tile_pixels_x + px) * c;

                    for (int ch = 0; ch < c; ++ch) {
                        hwc_data[dst_index + ch] = tile[src_index + ch];
                    }
                }
            }
        }
    }
}
}
