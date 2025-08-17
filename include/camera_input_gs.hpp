#pragma once
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <mutex>

class CameraInputAppsink {
public:
    enum class SourceType { UVC, CSI };

private:
    // GStreamer objects
    GstElement* pipeline_ = nullptr;
    GstElement* appsink_  = nullptr;

    // settings
    SourceType src_type_;
    std::string device_; // UVC only
    int width_, height_, fps_;

    // reusable buffers
    cv::Mat frame_bgr_;
    std::vector<float> chw_buffer_;

    // guard init_once
    static void ensure_gst_init() {
        static std::once_flag once;
        std::call_once(once, []{
            int argc = 0; char** argv = nullptr;
            gst_init(&argc, &argv);
        });
    }

    static std::string build_pipeline_string(SourceType type,
                                             const std::string& device,
                                             int width, int height, int fps)
    {
        if (type == SourceType::UVC) {
            // 低遅延：io-mode=2(mmap), バッファをためない, MJPEGならjpegdecを挟む
            // デバイスがMJPEGを出せない場合は 'image/jpeg' 部分を削って 'videoconvert' だけでもOK
            return
                "v4l2src device=" + device + " io-mode=2 ! "
                "image/jpeg,framerate=" + std::to_string(fps) + "/1,width=" + std::to_string(width) + ",height=" + std::to_string(height) + " ! "
                "jpegdec ! videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink name=sink max-buffers=1 drop=true sync=false";
        } else {
            // Jetson CSI（IMX219/IMX477など）：nvarguscamerasrc → NVMM → nvvidconv → BGR
            return
                "nvarguscamerasrc ! "
                "queue ! "
                "video/x-raw(memory:NVMM),width=" + std::to_string(width) +
                ",height=" + std::to_string(height) +
                ",framerate=" + std::to_string(fps) + "/1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink name=sink max-buffers=1 drop=true sync=false";
        }
    }

    void throw_if_error(bool ok, const std::string& msg) {
        if (!ok) throw std::runtime_error(msg);
    }

public:
    // UVC: device="/dev/video0", CSI: deviceは無視される
    CameraInputAppsink(SourceType type = SourceType::UVC,
                       const std::string& device = "/dev/video0",
                       int width = 640, int height = 480, int fps = 60)
        : src_type_(type), device_(device), width_(width), height_(height), fps_(fps)
    {
        ensure_gst_init();

        const std::string pipeline_str = build_pipeline_string(src_type_, device_, width_, height_, fps_);
        pipeline_ = gst_parse_launch(pipeline_str.c_str(), nullptr);
        if (!pipeline_) {
            throw std::runtime_error("GStreamer pipelineの構築に失敗しました: " + pipeline_str);
        }

        appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
        if (!appsink_) {
            gst_object_unref(pipeline_);
            throw std::runtime_error("appsink を取得できませんでした。");
        }

        // appsink 設定：シグナル不要、blocking pull
        gst_app_sink_set_emit_signals((GstAppSink*)appsink_, FALSE);
        gst_app_sink_set_drop((GstAppSink*)appsink_, TRUE);
        gst_app_sink_set_max_buffers((GstAppSink*)appsink_, 1);

        // 再生開始
        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            gst_object_unref(appsink_);
            gst_object_unref(pipeline_);
            throw std::runtime_error("GStreamer pipeline の再生開始に失敗しました。");
        }

        // 出力バッファ確保
        frame_bgr_.create(height_, width_, CV_8UC3);
        chw_buffer_.resize(3 * width_ * height_);

        std::cout << "==== Camera (appsink) Settings ====\n";
        std::cout << "Type   : " << (src_type_ == SourceType::UVC ? "UVC" : "CSI(nvargus)") << "\n";
        if (src_type_ == SourceType::UVC) std::cout << "Device : " << device_ << "\n";
        std::cout << "Width  : " << width_  << "\n";
        std::cout << "Height : " << height_ << "\n";
        std::cout << "FPS    : " << fps_    << "\n";
    }

    ~CameraInputAppsink() {
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }
        if (appsink_) {
            gst_object_unref(appsink_);
            appsink_ = nullptr;
        }
        if (pipeline_) {
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

    // 低遅延で1フレーム取得して CHW float32 (0..1) を返す
    // ※ ブロッキングで待機します。タイムアウトを付けたい場合は poll などを拡張してください。
    std::vector<float>& get_frame_chw() {
        if (!appsink_) {
            throw std::runtime_error("appsink が初期化されていません。");
        }

        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink_));
        if (!sample) {
            throw std::runtime_error("フレームの取得に失敗しました (sample=nullptr)。");
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps     = gst_sample_get_caps(sample);
        if (!buffer || !caps) {
            gst_sample_unref(sample);
            throw std::runtime_error("appsink sample から buffer/caps を取得できません。");
        }

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            throw std::runtime_error("GstBuffer の map に失敗しました。");
        }

        // GStreamerからBGRパック8bitで来る想定（video/x-raw,format=BGR）
        // map.data は連続BGR平面
        const int hw = width_ * height_;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(map.data);

        // OpenCV Mat にゼロコピーはできないので、先にMatに詰め替え（memcpy）
        // （アロケーションは再利用しているのでコストはコピーのみ）
        std::memcpy(frame_bgr_.data, src, static_cast<size_t>(hw * 3));

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        // BGR8 → CHW float32 (0..1)
        float* dst_b = chw_buffer_.data();
        float* dst_g = dst_b + hw;
        float* dst_r = dst_g + hw;

        const uint8_t* pix = frame_bgr_.ptr<uint8_t>(0);
        constexpr float k = 1.0f / 255.0f;
        for (int i = 0; i < hw; ++i) {
            // BGR
            dst_b[i] = pix[i * 3 + 0] * k;
            dst_g[i] = pix[i * 3 + 1] * k;
            dst_r[i] = pix[i * 3 + 2] * k;
        }
        return chw_buffer_;
    }
};
