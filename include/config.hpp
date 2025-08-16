#pragma once
#include <string>

#define MODEL_DIR "models/"
#define RESOURCE_DIR "resource/"

namespace config {
    constexpr const char* ENCODER_PATH = MODEL_DIR "encoder13_fp16_SP.engine";
    constexpr const char* DECODER_PATH = MODEL_DIR "decoder13_fp16_SP.engine";
    constexpr const char* IMAGE_PATH   = RESOURCE_DIR "8.jpg";
    //constexpr const char* CAMERA_HOST = "127.0.0.1";
    //constexpr const char* CAMERA_HOST = "10.211.55.11";
    //constexpr const char* CAMERA_HOST = "192.168.0.117";
    //constexpr const char* CAMERA_HOST = "192.168.0.177";
    constexpr const char* CAMERA_HOST = "192.168.0.170";
    constexpr uint16_t CAMERA_PORT     = 8004;
    constexpr size_t MAX_SAFE_UDP_SIZE = 1500;

    constexpr char* INPUT_SOURCE = "/dev/video0";
    constexpr int INPUT_W = 1280;
    constexpr int INPUT_H = 720;
    constexpr int INPUT_FPS = 30;

    constexpr int ENCODER_IN_C = 3;      // エンコーダー入力のチャンネル
    constexpr int ENCODER_IN_W = 1280;   // エンコーダー入力の幅
    constexpr int ENCODER_IN_H = 720;    // エンコーダー入力の高さ
    constexpr int ENCODER_OUT_C = 16;    // エンコーダー出力のチャンネル
    constexpr int ENCODER_OUT_W = 80;    // エンコーダー出力の幅
    constexpr int ENCODER_OUT_H = 45;    // エンコーダー出力の高さ

    constexpr int DECODER_IN_C = 16;     // デコーダー入力のチャンネル
    constexpr int DECODER_IN_W = 80;     // デコーダー入力の幅
    constexpr int DECODER_IN_H = 45;     // デコーダー入力の高さ
    constexpr int DECODER_OUT_C = 3;     // デコーダー出力のチャンネル
    constexpr int DECODER_OUT_W = 1280;  // デコーダー出力の幅
    constexpr int DECODER_OUT_H = 720;   // デコーダー出力の高さ

    constexpr int CHUNK_PIXEL = 88;
    constexpr int CHUNK_PIXEL_H = 8;
    constexpr int CHUNK_PIXEL_W = 10;
    constexpr int UDP_SO_SNDBUF = 64 * 1024;
    constexpr int UDP_SO_RCVBUF = 64 * 1024;
}
