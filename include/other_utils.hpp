#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <stdfloat>
#include <float_8.hpp>


#define EXPONENT_BITS 4

namespace other_utils {

// float16 → uint16_t ビット変換
inline std::vector<uint16_t> float16_to_u16(const std::vector<std::float16_t>& input) {
  std::vector<uint16_t> out;
  out.reserve(input.size());
  for (std::float16_t val : input) {
      uint16_t bits;
      std::memcpy(&bits, &val, sizeof(bits));
      out.push_back(bits);
  }
  return out;
}

// uint16_t → float16 復元
inline std::vector<std::float16_t> u16_to_float16(const std::vector<uint16_t>& input) {
  std::vector<std::float16_t> out;
  out.reserve(input.size());
  for (uint16_t bits : input) {
      std::float16_t val;
      std::memcpy(&val, &bits, sizeof(bits));
      out.push_back(val);
  }
  return out;
}

// float32 → uint32_t ビット変換
inline std::vector<std::uint32_t> float32_to_u32(const std::vector<float>& input) {
  std::vector<std::uint32_t> out;
  out.reserve(input.size());
  for (float val : input) {
      std::uint32_t bits;
      std::memcpy(&bits, &val, sizeof(bits));
      out.push_back(bits);
  }
  return out;
}

// uint32_t → float32 復元
inline std::vector<float> u32_to_float32(const std::vector<std::uint32_t>& input) {
  std::vector<float> out;
  out.reserve(input.size());
  for (std::uint32_t bits : input) {
      float val;
      std::memcpy(&val, &bits, sizeof(bits));
      out.push_back(val);
  }
  return out;
}

inline std::vector<uint8_t> float32_to_fp8_bytes(const std::vector<float>& input) {
  std::vector<uint8_t> out;
  out.reserve(input.size());
  for (float val : input) {
      float8_t f8 = float_to_float8(val);
      uint8_t bits;
      std::memcpy(&bits, &f8, sizeof(uint8_t));
      out.push_back(bits);
  }
  return out;
}

inline std::vector<float> fp8_bytes_to_float32(const std::vector<uint8_t>& input) {
  std::vector<float> out;
  out.reserve(input.size());
  for (uint8_t bits : input) {
      float8_t f8;
      std::memcpy(&f8, &bits, sizeof(uint8_t));
      float f32 = float8_to_float(f8);
      out.push_back(f32);
  }
  return out;
}


// 任意の浮動小数点型に対応した画像読み込み（RGB → [0.0, 1.0] 正規化）
template <typename T = float>
inline std::vector<T> load_rgbf(const std::string& path, int w, int h) {
    int c;
    uint8_t* img = stbi_load(path.c_str(), &w, &h, &c, 3);  // RGBとして読み込み
    if (!img) throw std::runtime_error("Failed to load image");

    std::vector<T> float_image(w * h * 3);
    for (int i = 0; i < w * h * 3; ++i) {
        float_image[i] = static_cast<T>(img[i]) / static_cast<T>(255.0f);
    }

    stbi_image_free(img);
    return float_image;
}
}