#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <stdfloat>

namespace quantizer {

// floatX → uint8 量子化（0.0〜1.0 → 0〜255）
template <typename T = float>
inline std::vector<uint8_t> quantize(const std::vector<T>& input) {
    std::vector<uint8_t> result;
    result.reserve(input.size());

    for (T val : input) {
        val = std::clamp(val, T(0.0), T(1.0));
        result.push_back(static_cast<uint8_t>(val * T(255.0) + T(0.5)));
    }
    return result;
}

// uint8 → floatX 逆量子化（0〜255 → 0.0〜1.0）
template <typename T = float>
inline std::vector<T> dequantize(const std::vector<uint8_t>& input) {
    std::vector<T> result;
    result.reserve(input.size());

    for (uint8_t val : input) {
        result.push_back(static_cast<T>(val) / T(255.0));
    }
    return result;
}
}
