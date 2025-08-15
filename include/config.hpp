#pragma once
#include <string>

#define MODEL_DIR "models/"
#define RESOURCE_DIR "resource/"

namespace config {
    constexpr const char* CAMERA_SOURCE = "/dev/video0";
    constexpr const char* ENCODER_PATH = MODEL_DIR "encoder.engine";
    constexpr const char* DECODER_PATH = MODEL_DIR "decoder.engine";
    constexpr const char* IMAGE_PATH   = RESOURCE_DIR "8.jpg";
    //constexpr const char* CAMERA_HOST = "127.0.0.1";
    constexpr const char* CAMERA_HOST = "10.211.55.11";
    //constexpr const char* CAMERA_HOST = "192.168.0.117";
    //constexpr const char* CAMERA_HOST = "192.168.0.177";
    //constexpr const char* CAMERA_HOST = "192.168.0.170";
    constexpr uint16_t CAMERA_PORT     = 8004;
    constexpr size_t MAX_SAFE_UDP_SIZE = 1500;
    constexpr int ENCODER_IN_C = 3;
    constexpr int ENCODER_IN_W = 1280;
    constexpr int ENCODER_IN_H = 720;
    constexpr int ENCODER_OUT_C = 16;  // エンコーダー出力のチャンネル
    constexpr int ENCODER_OUT_W = 80;  // エンコーダー出力の幅
    constexpr int ENCODER_OUT_H = 45;  // エンコーダー出力の高さ
    constexpr int CHUNK_PIXEL = 80;
    constexpr int CHUNK_PIXEL_H = 8;
    constexpr int CHUNK_PIXEL_W = 10;
    constexpr int FRAME_FPS = 30;
    constexpr int UDP_SO_SNDBUF = 64 * 1024;
    constexpr int UDP_SO_RCVBUF = 64 * 1024;
}
