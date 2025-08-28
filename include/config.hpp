#pragma once
#include <string>

#define MODEL_DIR "models/"
#define RESOURCE_DIR "resource/"

namespace config {
    // ============================
    // モデルパス
    // ============================
    constexpr const char* ENCODER_PATH = MODEL_DIR "encoder_int8.engine";
    constexpr const char* DECODER_PATH = MODEL_DIR "decoder_int8.engine";

    // ============================
    // 通信設定
    // ============================
    constexpr const char* CAMERA_HOST = "192.168.0.170";
    constexpr uint16_t CAMERA_PORT     = 8004;
    constexpr uint16_t SERVER_PORT     = 8004;
    constexpr size_t MAX_SAFE_UDP_SIZE = 1500;

    // ============================
    // 入力ソース情報
    // ============================
    constexpr const char* INPUT_SOURCE = "/dev/video0";
    constexpr int INPUT_W = 1280;
    constexpr int INPUT_H = 720;
    constexpr int INPUT_FPS = 60;

    // ============================
    // GStreamer パイプライン
    // ============================
    // UVC (USB, MJPEG)
    inline std::string UVC_GS_PIPELINE =
        "v4l2src device=" + std::string(INPUT_SOURCE) + " io-mode=2 ! "
        "image/jpeg,framerate=" + std::to_string(INPUT_FPS) + "/1,"
        "width=" + std::to_string(INPUT_W) + ",height=" + std::to_string(INPUT_H) + " ! "
        "jpegdec ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink name=sink max-buffers=1 drop=true sync=false";

    // CSI (Jetson カメラ, nvargus)
    inline std::string CSI_GS_PIPELINE =
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM),width=" + std::to_string(INPUT_W) +
        ",height=" + std::to_string(INPUT_H) +
        ",framerate=" + std::to_string(INPUT_FPS) + "/1 ! "
        "nvvidconv ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink name=sink max-buffers=1 drop=true sync=false";

    // ============================
    // エンコーダ / デコーダ形状
    // ============================
    constexpr int ENCODER_IN_C = 3;      // 入力C
    constexpr int ENCODER_IN_W = 1280;   // 入力W
    constexpr int ENCODER_IN_H = 720;    // 入力H
    constexpr int ENCODER_OUT_C = 16;    // 出力C
    constexpr int ENCODER_OUT_W = 80;    // 出力W
    constexpr int ENCODER_OUT_H = 45;    // 出力H

    constexpr int DECODER_IN_C = 16;     // 入力C
    constexpr int DECODER_IN_W = 80;     // 入力W
    constexpr int DECODER_IN_H = 45;     // 入力H
    constexpr int DECODER_OUT_C = 3;     // 出力C
    constexpr int DECODER_OUT_W = 1280;  // 出力W
    constexpr int DECODER_OUT_H = 720;   // 出力H

    // ============================
    // チャンク / UDP 設定
    // ============================
    constexpr int CHUNK_PIXEL   = 88;
    constexpr int CHUNK_PIXEL_H = 8;
    constexpr int CHUNK_PIXEL_W = 10;
    constexpr int UDP_SO_SNDBUF = 256 * 1024;
    constexpr int UDP_SO_RCVBUF = 256 * 1024;

    // ============================
    // その他
    // ============================
    constexpr float FRAME_SKIP_THRESHOLD = 0.4f;
}
