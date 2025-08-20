#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstdint>

class PixelShuffler {
public:
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
