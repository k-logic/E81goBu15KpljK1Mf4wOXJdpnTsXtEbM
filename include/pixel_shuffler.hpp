#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstdint>
#include <PoissonGenerator.h>

class PixelShuffler {
public:
    /*
    // stride shuffle
    PixelShuffler(int h, int w, int c, int step = -1)
        : height(h), width(w), channels(c), num_pixels(h * w)
    {
        if (num_pixels <= 1) {
            throw std::invalid_argument("num_pixels must be > 1");
        }

        // stepが指定されていない場合は、適当に num_pixels/2+1 にする
        step = num_pixels / 2 + 1;

        // stepとnum_pixelsが互いに素であることを確認
        if (std::gcd(step, num_pixels) != 1) {
            throw std::invalid_argument("step must be coprime with num_pixels");
        }

        shuffle_table.resize(num_pixels);
        for (int i = 0; i < num_pixels; i++) {
            shuffle_table[i] = (i * step) % num_pixels;
        }

        inverse_table.resize(num_pixels);
        for (int i = 0; i < num_pixels; i++) {
            inverse_table[shuffle_table[i]] = i;
        }
    }
    */

    // ノーマルランダム
    PixelShuffler(int h, int w, int c, unsigned seed = 1234)
        : height(h), width(w), channels(c), num_pixels(h * w) 
    {
        shuffle_table.resize(num_pixels);
        std::iota(shuffle_table.begin(), shuffle_table.end(), 0);

        std::mt19937 rng(seed);
        std::shuffle(shuffle_table.begin(), shuffle_table.end(), rng);

        inverse_table.resize(num_pixels);
        for (int i = 0; i < num_pixels; i++) {
            inverse_table[shuffle_table[i]] = i;
        }
    }

    // Poisson分布(ブルーノイズ)
    /*
    PixelShuffler(int h, int w, int c, float minDist = -1.0f, unsigned seed = 1234)
        : height(h), width(w), channels(c), num_pixels(h * w) 
    {
        if (minDist <= 0.0f) {
            minDist = 1.0f / std::max(width, height); // デフォルト: 1ピクセル相当
        }

        PoissonGenerator::DefaultPRNG rng(seed);

        auto points = PoissonGenerator::generatePoissonPoints(
            num_pixels,   // 最大点数
            rng,
            false,
            30,
            minDist
        );

        shuffle_table.reserve(num_pixels);

        for (auto& p : points) {
            int x = static_cast<int>(p.x * width);
            int y = static_cast<int>(p.y * height);
            if (x >= 0 && x < width && y >= 0 && y < height) {
                shuffle_table.push_back(y * width + x);
            }
        }

        // 不足を埋める
        if ((int)shuffle_table.size() < num_pixels) {
            std::vector<int> rest(num_pixels);
            std::iota(rest.begin(), rest.end(), 0);
            for (int idx : shuffle_table) rest[idx] = -1;
            for (int idx : rest) {
                if (idx >= 0) shuffle_table.push_back(idx);
            }
        }

        inverse_table.resize(num_pixels);
        for (int i = 0; i < num_pixels; i++) {
            inverse_table[shuffle_table[i]] = i;
        }
    }
    */


    // シャッフル
    void shuffle(const std::vector<uint8_t>& src,
                 std::vector<uint8_t>& dst) const {
        dst.resize(src.size());
        for (int i = 0; i < num_pixels; i++) {
            int j = shuffle_table[i];
            for (int c = 0; c < channels; c++) {
                dst[j * channels + c] = src[i * channels + c];
            }
        }
    }

    // 逆シャッフル
    void inverse(const std::vector<uint8_t>& src,
                 std::vector<uint8_t>& dst) const {
        dst.resize(src.size());
        for (int j = 0; j < num_pixels; j++) {
            int i = inverse_table[j];
            for (int c = 0; c < channels; c++) {
                dst[i * channels + c] = src[j * channels + c];
            }
        }
    }

private:
    int height, width, channels, num_pixels;
    std::vector<int> shuffle_table;
    std::vector<int> inverse_table;
};
