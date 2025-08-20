#if !defined(USE_TENSORRT) && !defined(USE_TFLITE)
    #error "You must define either -DUSE_TENSORRT or -DUSE_TFLITE"
#endif

#define FMT_HEADER_ONLY
// 外部ライブラリ
#include <fmt/core.h>
#include <opencv2/opencv.hpp>
// 自作ライブラリ
#include <config.hpp>
#include <packet.hpp>
#include <chunker.hpp>
#include <other_utils.hpp>
#include <debug_utils.hpp>
#include <udp_sender.hpp>
#include <camera_input2.hpp>
#include <camera_input_gs.hpp>

#if defined(USE_TENSORRT)
#include <IModelExecutor.hpp>
#include <TensorRTExecutor.hpp>
#include <cuda_runtime.h>
#endif

#if defined(USE_TFLITE)
#include <IModelExecutor.hpp>
#include <TFLiteExecutor.hpp>
#endif

using namespace config;

void send_chunks(asio::io_context& io, UdpSender& sender, int frame_id, const std::vector<std::vector<uint8_t>>& chunks) {
    std::vector<size_t> indices(chunks.size());
    std::iota(indices.begin(), indices.end(), 0);
    static std::mt19937 rng(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t i : indices) {
        std::vector<uint8_t> uint8_data = chunks[i];
        packet::packet_header header{
            static_cast<uint16_t>(frame_id),
            static_cast<uint16_t>(i),
            static_cast<uint16_t>(chunks.size()),
            0
        };
        std::vector<uint8_t> packet = packet::make_packet_u8(header, uint8_data);
        if (packet.size() > MAX_SAFE_UDP_SIZE) {
            std::cerr << fmt::format("Packet too large ({} bytes), skipping.\n", packet.size());
            continue;
        }

        std::cout << fmt::format("frame {} chunk {} size: {} bytes\n", frame_id, i, packet.size());
        sender.send_sync(packet);
    }
}

int main() {
    try {
        asio::io_context io;
        UdpSender sender;
        sender.init_sync(io, CAMERA_HOST, CAMERA_PORT);

        //CameraInput camera(INPUT_SOURCE, INPUT_W, INPUT_H, INPUT_FPS);
        // CSI:UVC
        CameraInputAppsink camera(CameraInputAppsink::SourceType::CSI, INPUT_SOURCE, INPUT_W, INPUT_H, INPUT_FPS);

        std::unique_ptr<IModelExecutor> encoder_model;

        // liteRT用
        #if defined(USE_TFLITE)
            encoder_model = std::make_unique<TFLiteExecutor>();
            encoder_model->load(ENCODER_PATH);
        #endif

        // TensorRT用
        #if defined(USE_TENSORRT)
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            encoder_model = std::make_unique<TensorRTExecutor>(stream);
            encoder_model->load(ENCODER_PATH);
        #endif

        std::vector<float> encoded;
        std::vector<std::vector<uint8_t>> chunks;
        int frame_id = 0;

        while (true) {
            int key = cv::waitKey(1);
            if (key == 'q') break;
            
            // ===== 時間計測用 =====
            auto t0 = std::chrono::high_resolution_clock::now();
            auto t_prev = t0;

            // 1. カメラ取得
            std::vector<float> input = camera.get_frame_chw();
            auto t1 = std::chrono::high_resolution_clock::now();
            
            if (input.empty()) {
                std::cerr << "[ERROR] Failed to capture frame.\n";
                continue;
            }
            
            // 2. 推論
            encoder_model->run(input, encoded);
            auto t2 = std::chrono::high_resolution_clock::now();

            // float32->float8に量子化
            std::vector<uint8_t> encoded_fp8 = other_utils::float32_to_fp8(encoded);
            
            // 3. チャンク分割
            chunker::chunk_by_tiles_hwc(
                encoded_fp8,
                ENCODER_OUT_C,
                ENCODER_OUT_H,
                ENCODER_OUT_W,
                CHUNK_PIXEL_W,
                CHUNK_PIXEL_H,
                chunks
            );
            auto t3 = std::chrono::high_resolution_clock::now();
            
            // 4. UDP送信
            send_chunks(io, sender, frame_id, chunks);
            auto t4 = std::chrono::high_resolution_clock::now();
            
            // ===== 各区間の時間（ms）を計算 =====
            auto ms_getframe = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t_prev).count();
            auto ms_encode   = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            auto ms_chunk    = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
            auto ms_send     = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
            auto ms_total    = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t0).count();

            fmt::print("[TIME] get_frame: {} ms | encode: {} ms | chunk: {} ms | send: {} ms | total: {} ms\n",
                       ms_getframe, ms_encode, ms_chunk, ms_send, ms_total);
            
            frame_id++;
        }
        return 0;

    } catch (const std::exception& e) {
        std::cerr << fmt::format("[ERROR] {}\n",  e.what());
        return -1;
    }
}
