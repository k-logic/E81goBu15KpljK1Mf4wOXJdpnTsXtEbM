#pragma once

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>

namespace image_display {

// シャープ化設定
#ifndef SHARPEN_ENABLE
#define SHARPEN_ENABLE 1   // 0で無効化
#endif

#ifndef SHARPEN_METHOD_USM
#define SHARPEN_METHOD_USM 1  // 1: アンシャープマスク / 0: 3x3 シャープカーネル
#endif

// 調整用パラメータ
static double g_usm_sigma   = 1.0;  // ぼかし強さ
static double g_usm_amount  = 0.6;  // 強調量
static int    g_kernel_ksz  = 0;    // 0ならsigma優先

// ===== CHW入力（C=3前提、BGR想定） =====
inline void display_decoded_image_chw(const float* chw, int c, int h, int w) {
    static auto last_time = std::chrono::high_resolution_clock::now();
    static cv::Mat image_f32;
    static cv::Mat image_u8;

    const size_t hw = static_cast<size_t>(h) * static_cast<size_t>(w);

    // 出力バッファ確保
    if (image_f32.empty() || image_f32.rows != h || image_f32.cols != w) {
        image_f32 = cv::Mat(h, w, CV_32FC3);
    }

    float* dst = reinterpret_cast<float*>(image_f32.data);

    // CHW -> HWC (BGR)
    for (size_t i = 0; i < hw; ++i) {
        dst[i * 3 + 0] = chw[0 * hw + i]; // B
        dst[i * 3 + 1] = chw[1 * hw + i]; // G
        dst[i * 3 + 2] = chw[2 * hw + i]; // R
    }

    // float[0..1] -> uint8
    if (image_u8.empty() || image_u8.rows != h || image_u8.cols != w) {
        image_u8 = cv::Mat(h, w, CV_8UC3);
    }
    image_f32.convertTo(image_u8, CV_8UC3, 255.0);

#if SHARPEN_ENABLE
#if SHARPEN_METHOD_USM
    {
        static cv::Mat blur_u8;
        if (blur_u8.empty() || blur_u8.rows != h || blur_u8.cols != w) {
            blur_u8 = cv::Mat(h, w, CV_8UC3);
        }
        cv::GaussianBlur(image_u8, blur_u8,
                         (g_kernel_ksz > 0) ? cv::Size(g_kernel_ksz, g_kernel_ksz) : cv::Size(0, 0),
                         g_usm_sigma, g_usm_sigma, cv::BORDER_REPLICATE);
        cv::addWeighted(image_u8, 1.0 + g_usm_amount, blur_u8, -g_usm_amount, 0.0, image_u8);
    }
#else
    {
        static cv::Mat kernel = (cv::Mat_<float>(3,3) <<
             0, -1,  0,
            -1,  5, -1,
             0, -1,  0
        );
        cv::filter2D(image_u8, image_u8, -1, kernel, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
    }
#endif
#endif

    // FPS表示
    auto now = std::chrono::high_resolution_clock::now();
    float fps = 1000.0f / std::chrono::duration<float, std::milli>(now - last_time).count();
    last_time = now;

    static char fps_buf[32];
    std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
    cv::putText(image_u8, fps_buf, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2, cv::LINE_AA);

    cv::imshow("Decoded", image_u8);
    cv::waitKey(1);
}

// ===== HWC入力（C=3前提、BGRまたはRGB） =====
// #define HWC_IS_RGB でRGB→BGR変換
inline void display_decoded_image_hwc(const float* hwc, int c, int h, int w) {
    static auto last_time = std::chrono::high_resolution_clock::now();
    static cv::Mat image_u8;

    if (!hwc || c != 3) return;

    // HWCをそのままMatにラップ
    cv::Mat image_f32(h, w, CV_32FC3, const_cast<float*>(hwc));

    // float[0..1] -> uint8
    if (image_u8.empty() || image_u8.rows != h || image_u8.cols != w) {
        image_u8 = cv::Mat(h, w, CV_8UC3);
    }
    image_f32.convertTo(image_u8, CV_8UC3, 255.0);

#if defined(HWC_IS_RGB)
    cv::cvtColor(image_u8, image_u8, cv::COLOR_RGB2BGR);
#endif

#if SHARPEN_ENABLE
#if SHARPEN_METHOD_USM
    {
        static cv::Mat blur_u8;
        if (blur_u8.empty() || blur_u8.rows != h || blur_u8.cols != w) {
            blur_u8 = cv::Mat(h, w, CV_8UC3);
        }
        cv::GaussianBlur(image_u8, blur_u8,
                         (g_kernel_ksz > 0) ? cv::Size(g_kernel_ksz, g_kernel_ksz) : cv::Size(0, 0),
                         g_usm_sigma, g_usm_sigma, cv::BORDER_REPLICATE);
        cv::addWeighted(image_u8, 1.0 + g_usm_amount, blur_u8, -g_usm_amount, 0.0, image_u8);
    }
#else
    {
        static cv::Mat kernel = (cv::Mat_<float>(3,3) <<
             0, -1,  0,
            -1,  5, -1,
             0, -1,  0
        );
        cv::filter2D(image_u8, image_u8, -1, kernel, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
    }
#endif
#endif

    // FPS表示
    auto now = std::chrono::high_resolution_clock::now();
    float fps = 1000.0f / std::chrono::duration<float, std::milli>(now - last_time).count();
    last_time = now;

    static char fps_buf[32];
    std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
    cv::putText(image_u8, fps_buf, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2, cv::LINE_AA);

    cv::imshow("Decoded", image_u8);
    cv::waitKey(1);
}
}