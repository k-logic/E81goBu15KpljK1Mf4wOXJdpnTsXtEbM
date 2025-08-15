#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <random>
#include <span>

namespace chunker {
// CHW形式のエンコード出力を「ピクセルN個単位」でチャンク化
inline void chunk_by_pixels_chw(
    const std::vector<float>& chw_data,     // 入力: CHW (C平面ごと連続) サイズ= c*h*w
    int c, int h, int w,                    // チャンネル数・高さ・幅
    int pixels_per_chunk,                   // 1チャンクに含めるピクセル数
    std::vector<std::vector<float>>& chunks // 出力: 再利用するチャンク配列（HWC並び）
) {
    if (c <= 0 || h <= 0 || w <= 0 || pixels_per_chunk <= 0) {
        throw std::invalid_argument("c, h, w, pixels_per_chunk must be > 0");
    }

    const int hw = h * w;
    const size_t expected = static_cast<size_t>(hw) * static_cast<size_t>(c);
    if (chw_data.size() != expected) {
        throw std::invalid_argument("chw_data size mismatch: expected c*h*w elements");
    }

    // 総チャンク数を決めて外側ベクタを再利用
    const int num_chunks = (hw + pixels_per_chunk - 1) / pixels_per_chunk;
    chunks.resize(num_chunks);

    // 各チャンネル平面の先頭ポインタ（ループ内の掛け算/加算を減らす）
    // cが小さい（例:3）前提なら、ここは軽いコスト
    std::vector<const float*> planes;
    planes.reserve(static_cast<size_t>(c));
    for (int ch = 0; ch < c; ++ch) {
        planes.push_back(chw_data.data() + static_cast<size_t>(ch) * hw);
    }

    // チャンクごとにHWC並びで書き出し（再確保なし、push_backなし）
    int start_pixel = 0;
    for (int i = 0; i < num_chunks; ++i, start_pixel += pixels_per_chunk) {
        const int end_pixel = std::min(start_pixel + pixels_per_chunk, hw);
        const int npix = end_pixel - start_pixel;
        const size_t out_elems = static_cast<size_t>(npix) * static_cast<size_t>(c);

        auto& out = chunks[i];
        out.resize(out_elems); // 内側も再利用（capacityが足りなければ初回のみ再確保）
        float* dst = out.data();

        // [start_pixel, end_pixel) の各ピクセルについて HWC順に詰める
        // dst は [p=0..npix-1] * c のレイアウト
        // CHW読み込みは planes[ch][p_global]（p_global = start_pixel + p）
        size_t out_idx = 0;
        for (int p = 0; p < npix; ++p) {
            const int p_global = start_pixel + p; // = base_hw
            for (int ch = 0; ch < c; ++ch) {
                dst[out_idx++] = planes[ch][p_global];
            }
        }
    }
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
inline void reconstruct_from_chunks_chw(
    const std::vector<std::vector<float>>& chunks,
    float* chw_data,
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

// chunks(=HWCでピクセル連続) → HWC一次元配列 へ再構築
inline void reconstruct_from_chunks_hwc(
    const std::vector<std::vector<float>>& chunks, // ソート済み・欠損補完済み
    float* hwc_data,
    int c, int h, int w
) {
    const int hw = h * w;
    int pixel_index = 0;

    for (const auto& chunk : chunks) {
        const int num_pixels = static_cast<int>(chunk.size()) / c;
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
    const std::vector<float>& hwc,       // 入力 HWC 配列
    int c,                               // チャンネル数
    int h,                               // 高さ
    int w,                               // 幅
    int tile_w,                          // タイル幅（ピクセル単位）
    int tile_h,                          // タイル高さ（ピクセル単位）
    std::vector<std::vector<float>>& chunks // 出力チャンク配列（再利用）
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

    const float* base = hwc.data();

    for (int ty = 0; ty < tiles_y; ++ty) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            int start_x = tx * tile_w;
            int start_y = ty * tile_h;
            int end_x   = std::min(start_x + tile_w, w);
            int end_y   = std::min(start_y + tile_h, h);

            // タイル内のピクセルを収集
            std::vector<float> chunk;
            chunk.reserve((end_y - start_y) * (end_x - start_x) * c);

            for (int y = start_y; y < end_y; ++y) {
                const float* row_ptr = base + (y * w * c);
                for (int x = start_x; x < end_x; ++x) {
                    const float* pixel_ptr = row_ptr + (x * c);
                    chunk.insert(chunk.end(), pixel_ptr, pixel_ptr + c);
                }
            }
            chunks.push_back(std::move(chunk));
        }
    }
}


inline void reconstruct_from_tiles_hwc(
    const std::vector<std::vector<float>>& chunks, // タイルごとのデータ
    const std::vector<bool>& received_flags,       // タイル到着フラグ
    float* hwc_data,                               // 出力HWCバッファ
    int c, int h, int w,                           // チャンネル数, 高さ, 幅
    int tile_w, int tile_h                         // タイル幅・高さ（ピクセル単位）
) {
    const int tiles_x = (w + tile_w - 1) / tile_w;  // 横方向タイル数
    const int tiles_y = (h + tile_h - 1) / tile_h;  // 縦方向タイル数
    const int expected_tile_count = tiles_x * tiles_y;

    if (chunks.size() != static_cast<size_t>(expected_tile_count) ||
        received_flags.size() != static_cast<size_t>(expected_tile_count)) {
        std::cerr << "[WARN] Tile count mismatch: expected "
                  << expected_tile_count
                  << " but got chunks=" << chunks.size()
                  << ", flags=" << received_flags.size()
                  << " — missing tiles will be filled black.\n";
    }

    // 出力バッファを黒（0.0f）で初期化
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
