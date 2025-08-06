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
#include <camera_input.hpp>

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

void send_chunks(asio::io_context& io, UdpSender& sender, int frame_id, const std::vector<std::vector<float>>& chunks) {
    std::vector<size_t> indices(chunks.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    for (size_t i : indices) {
        std::cout << fmt::format("ーーーーーーーーーーーーーーーーーーーーーーーーー\n");
        std::vector<uint8_t> uint8_data = other_utils::float32_to_fp8_bytes(chunks[i]);
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
        std::cout << fmt::format("packet size: {} bytes for frame id: {} (chunk_id: {})\n", packet.size(), frame_id, i);
        sender.send_sync(packet);
    }
}

int main() {
    try {
        asio::io_context io;
        UdpSender sender;
        sender.init_sync(io, CAMERA_HOST, CAMERA_PORT);

        CameraInput camera(0, IMAGE_W, IMAGE_H, FRAME_FPS);

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
        int frame_id = 0;
        while (true) {
            // キー入力
            int key = cv::waitKey(10);
            if (key == 'q') break;
            std::vector<float> input = camera.get_frame_chw();
            if (input.empty()) {
                std::cerr << "[ERROR] Failed to capture frame.\n";
                continue;
            }
            
            encoder_model->run(input, encoded);
            
            std::cout << fmt::format("Encoded size: {} byte\n", encoded.size() * 4);
            std::cout << fmt::format("Output size: {} byte\n", encoded.size());    
            std::vector<std::vector<float>> chunks = chunker::chunk_by_pixels<float>(encoded, CHUNK_C, CHUNK_H, CHUNK_W, CHUNK_PIXEL);
            debug_utils::print_chunk_info(chunks, CHUNK_H, CHUNK_W);
            std::cout << fmt::format("Total Chunk: {}\n", chunks.size());  
            send_chunks(io, sender, frame_id, chunks);
            frame_id++;
        }
        return 0;

    } catch (const std::exception& e) {
        std::cerr << fmt::format("[ERROR] {}\n",  e.what());
        return -1;
    }
}