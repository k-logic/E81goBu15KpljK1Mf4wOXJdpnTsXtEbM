#pragma once
#include <string>

#define MODEL_DIR "models/"
#define RESOURCE_DIR "resource/"

namespace config {
    constexpr const char* ENCODER_PATH = MODEL_DIR "encoder.tflite";
    constexpr const char* DECODER_PATH = MODEL_DIR "decoder.tflite";
    constexpr const char* IMAGE_PATH   = RESOURCE_DIR "8.jpg";
    //constexpr const char* CAMERA_HOST = "127.0.0.1";
    constexpr const char* CAMERA_HOST = "10.211.55.11";
    //constexpr const char* CAMERA_HOST = "192.168.0.117";
    //constexpr const char* CAMERA_HOST = "192.168.0.177";
    //constexpr const char* CAMERA_HOST = "192.168.0.170";
    constexpr uint16_t CAMERA_PORT     = 8004;
    constexpr size_t MAX_SAFE_UDP_SIZE = 1500;
    constexpr int IMAGE_C = 3;
    constexpr int IMAGE_W = 1280;
    constexpr int IMAGE_H = 720;
    constexpr int CHUNK_C = 16;
    constexpr int CHUNK_W = 80;
    constexpr int CHUNK_H = 45;
    constexpr int CHUNK_PIXEL = 80;
    constexpr int FRAME_FPS = 30;
    constexpr int BUFF_FRAME = 1;
    constexpr bool REALTIME_MODE = true;
}
