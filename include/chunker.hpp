#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <stdfloat>
#include <random>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

namespace chunker {

// 任意の型に対応したチャンク分割関数
template <typename T = float>
inline std::vector<std::vector<T>> split_chunks(
    const std::vector<T>& float_image,
    int width, int height,
    int CHUNK_W, int CHUNK_H
) {
    std::vector<std::vector<T>> chunks;

    for (int by = 0; by + CHUNK_H <= height; by += CHUNK_H) {
        for (int bx = 0; bx + CHUNK_W <= width; bx += CHUNK_W) {
            std::vector<T> chunk;
            for (int cy = 0; cy < CHUNK_H; ++cy) {
                for (int cx = 0; cx < CHUNK_W; ++cx) {
                    int pixel_index = ((by + cy) * width + (bx + cx)) * 3;
                    chunk.push_back(float_image[pixel_index + 0]); // R
                    chunk.push_back(float_image[pixel_index + 1]); // G
                    chunk.push_back(float_image[pixel_index + 2]); // B
                }
            }
            chunks.push_back(std::move(chunk));
        }
    }

    return chunks;
}

// CHW形式のfloat_imageに対応したチャンク分割
template <typename T = float>
inline std::vector<std::vector<T>> split_chunks_chw(
    const std::vector<T>& float_image,
    int width, int height,
    int CHUNK_W, int CHUNK_H,
    int channels = 16
) {
    std::vector<std::vector<T>> chunks;
    const int hw = height * width;

    for (int by = 0; by + CHUNK_H <= height; by += CHUNK_H) {
        for (int bx = 0; bx + CHUNK_W <= width; bx += CHUNK_W) {
            std::vector<T> chunk;
            for (int cy = 0; cy < CHUNK_H; ++cy) {
                for (int cx = 0; cx < CHUNK_W; ++cx) {
                    int y = by + cy;
                    int x = bx + cx;
                    for (int c = 0; c < channels; ++c) {
                        int idx = c * hw + y * width + x;
                        chunk.push_back(float_image[idx]);
                    }
                }
            }
            chunks.push_back(std::move(chunk));
        }
    }

    return chunks;
}


// CHW形式のエンコード出力を「ピクセルN個単位」でチャンク化
template <typename T = float>
std::vector<std::vector<T>> chunk_by_pixels(
    const std::vector<T>& chw_data,
    int c, int h, int w,
    int pixels_per_chunk = 16  // デフォルトは16ピクセルごとに1チャンク
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
    std::vector<std::vector<T>> result(chunks.size());  // 元のサイズと同じ

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int kept = 0, dropped = 0;

    for (size_t i = 0; i < chunks.size(); ++i) {
        if (dis(gen) > drop_rate) {
            result[i] = chunks[i];  // 元のチャンクをそのまま
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



template<typename T = float>
inline std::vector<T> merge_chunks(
    const std::unordered_map<int, std::vector<T>>& chunks,
    int width, int height,
    int CHUNK_W, int CHUNK_H
) {
    constexpr int CHANNELS = 3;
    std::vector<T> restored_image(width * height * CHANNELS, static_cast<T>(0));
    const std::vector<T> black_chunk(CHUNK_W * CHUNK_H * CHANNELS, static_cast<T>(0));

    int chunk_cols = width / CHUNK_W;
    int chunk_rows = height / CHUNK_H;

    for (int i = 0; i < chunk_cols * chunk_rows; ++i) {
        auto it = chunks.find(i);
        const std::vector<T>& source =
            (it != chunks.end() && !it->second.empty())
            ? it->second : black_chunk;

        int chunk_x = (i % chunk_cols) * CHUNK_W;
        int chunk_y = (i / chunk_cols) * CHUNK_H;

        for (int cy = 0; cy < CHUNK_H; ++cy)
            for (int cx = 0; cx < CHUNK_W; ++cx)
                for (int ch = 0; ch < CHANNELS; ++ch) {
                    int dst_idx = ((chunk_y + cy) * width + (chunk_x + cx)) * CHANNELS + ch;
                    int src_idx = (cy * CHUNK_W + cx) * CHANNELS + ch;
                    restored_image[dst_idx] = source[src_idx];
                }
    }

    return restored_image;
}


template <typename T = float>
inline std::vector<std::vector<T>> split_chunks_general(
    const std::vector<T>& data,
    int channels, int width, int height,
    int CHUNK_W, int CHUNK_H
) {
    std::vector<std::vector<T>> chunks;

    for (int by = 0; by + CHUNK_H <= height; by += CHUNK_H) {
        for (int bx = 0; bx + CHUNK_W <= width; bx += CHUNK_W) {
            std::vector<T> chunk;
            for (int cy = 0; cy < CHUNK_H; ++cy) {
                for (int cx = 0; cx < CHUNK_W; ++cx) {
                    int base = ((by + cy) * width + (bx + cx)) * channels;
                    for (int ch = 0; ch < channels; ++ch) {
                        chunk.push_back(data[base + ch]);
                    }
                }
            }
            chunks.push_back(std::move(chunk));
        }
    }

    return chunks;
}



}
