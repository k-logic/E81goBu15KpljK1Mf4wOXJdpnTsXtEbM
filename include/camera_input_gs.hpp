#pragma once
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <mutex>
#include <cstring>

class CameraInputAppsink {
private:
    GstElement* pipeline_ = nullptr;
    GstElement* appsink_  = nullptr;

    std::string pipeline_str_;
    int width_  = 0;
    int height_ = 0;

    std::vector<float> chw_buffer_;

    static void ensure_gst_init() {
        static std::once_flag once;
        std::call_once(once, []{
            int argc = 0; char** argv = nullptr;
            gst_init(&argc, &argv);
        });
    }

public:
    explicit CameraInputAppsink(const std::string& pipeline_str)
        : pipeline_str_(pipeline_str)
    {
        ensure_gst_init();

        pipeline_ = gst_parse_launch(pipeline_str_.c_str(), nullptr);
        if (!pipeline_) {
            throw std::runtime_error("GStreamer pipelineの構築に失敗しました: " + pipeline_str_);
        }

        appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
        if (!appsink_) {
            gst_object_unref(pipeline_);
            throw std::runtime_error("appsink を取得できませんでした。");
        }

        gst_app_sink_set_emit_signals((GstAppSink*)appsink_, FALSE);
        gst_app_sink_set_drop((GstAppSink*)appsink_, TRUE);
        gst_app_sink_set_max_buffers((GstAppSink*)appsink_, 1);

        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            gst_object_unref(appsink_);
            gst_object_unref(pipeline_);
            throw std::runtime_error("GStreamer pipeline の再生開始に失敗しました。");
        }

        // 最初のサンプルで解像度を取得
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink_));
        if (!sample) throw std::runtime_error("初回フレームの取得に失敗しました。");

        GstCaps* caps = gst_sample_get_caps(sample);
        if (!caps) {
            gst_sample_unref(sample);
            throw std::runtime_error("caps を取得できませんでした。");
        }

        GstStructure* structure = gst_caps_get_structure(caps, 0);
        gst_structure_get_int(structure, "width", &width_);
        gst_structure_get_int(structure, "height", &height_);

        gst_sample_unref(sample);

        if (width_ == 0 || height_ == 0) {
            throw std::runtime_error("解像度を caps から取得できませんでした。");
        }

        chw_buffer_.resize(3 * width_ * height_);

        std::cout << "==== Camera (appsink) Settings ====\n";
        std::cout << "Pipeline: " << pipeline_str_ << "\n";
        std::cout << "Width   : " << width_  << "\n";
        std::cout << "Height  : " << height_ << "\n";
    }

    ~CameraInputAppsink() {
        if (pipeline_) gst_element_set_state(pipeline_, GST_STATE_NULL);
        if (appsink_)  gst_object_unref(appsink_);
        if (pipeline_) gst_object_unref(pipeline_);
    }

    std::vector<float>& get_frame_chw() {
        if (!appsink_) throw std::runtime_error("appsink が初期化されていません。");

        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink_));
        if (!sample) throw std::runtime_error("フレームの取得に失敗しました。");

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (!buffer) {
            gst_sample_unref(sample);
            throw std::runtime_error("appsink sample から buffer を取得できません。");
        }

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            throw std::runtime_error("GstBuffer の map に失敗しました。");
        }

        const int hw = width_ * height_;
        const uint8_t* pix = reinterpret_cast<const uint8_t*>(map.data);

        float* dst_b = chw_buffer_.data();
        float* dst_g = dst_b + hw;
        float* dst_r = dst_g + hw;

        constexpr float k = 1.0f / 255.0f;
        for (int i = 0; i < hw; ++i) {
            dst_b[i] = pix[i * 3 + 0] * k;
            dst_g[i] = pix[i * 3 + 1] * k;
            dst_r[i] = pix[i * 3 + 2] * k;
        }

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        return chw_buffer_;
    }
};
