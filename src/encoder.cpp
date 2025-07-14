#include <tflite_model.hpp>

#include <packet.hpp>
#include <chunker.hpp>
#include <other_utils.hpp>
#include <debug_utils.hpp>
#include <udp_sender.hpp>

#include <opencv2/opencv.hpp>

#define RESOURCE_DIR "resource/"
#define MODEL_DIR "models/"

const std::string ENCODER_PATH = MODEL_DIR "encoder.tflite";
const std::string DECODER_PATH = MODEL_DIR "decoder.tflite";
const std::string IMAGE_PATH   = RESOURCE_DIR "8.jpg";
const std::string CAMERA_HOST = "127.0.0.1";
const uint16_t CAMERA_PORT = 8004;
const size_t MAX_SAFE_UDP_SIZE = 1500;
const int IMAGE_C = 3;
const int IMAGE_W = 1280;
const int IMAGE_H = 720;
const int CHUNK_C = 16;
const int CHUNK_W = 80;
const int CHUNK_H = 45;
const int CHUNK_PIXEL = 88;
const int FRAME_FPS = 30;
const bool REALTIME_MODE = true;

cv::VideoCapture init_camera() {
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, IMAGE_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, IMAGE_H);
    cap.set(cv::CAP_PROP_FPS, FRAME_FPS);
    if (!cap.isOpened()) {
        throw std::runtime_error("カメラを開けませんでした。");
    }
    // カメラウォームアップ
    cv::Mat dummy;
    for (int i = 0; i < 10; ++i) {
        cap >> dummy;
        cv::waitKey(30);
    }

    return cap;
}

std::vector<float> convert_to_chw(cv::Mat& frame) {
    frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
    std::vector<float> input;
    input.reserve(IMAGE_C * IMAGE_H * IMAGE_W);
    for (int c = 0; c < IMAGE_C; ++c)
        for (int h = 0; h < IMAGE_H; ++h)
            for (int w = 0; w < IMAGE_W; ++w)
                input.push_back(frame.at<cv::Vec3f>(h, w)[c]);
    return input;
}

void send_chunks(asio::io_context& io, UdpSender& sender, int frame_id, const std::vector<std::vector<float>>& chunks) {
    std::vector<size_t> indices(chunks.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    for (size_t i : indices) {
        std::cout << std::format("ーーーーーーーーーーーーーーーーーーーーーーーーー\n");
        std::vector<uint8_t> uint8_data = other_utils::float32_to_fp8_bytes(chunks[i]);
        std::vector<uint8_t> packet = packet::make_packet_u8(frame_id, i, chunks.size(), 0, uint8_data);
        if (packet.size() > MAX_SAFE_UDP_SIZE) {
            std::cerr << std::format("Packet too large ({} bytes), skipping.\n", packet.size());
            continue;
        }
        std::cout << std::format("packet size: {} bytes for frame id: {} (chunk_id: {})\n", packet.size(), frame_id, i);
        sender.send_sync(packet);
    }
}

int main() {
    try {
        lite::Encoder encoder;
        encoder.load(ENCODER_PATH);

        asio::io_context io;
        UdpSender sender;
        sender.init_sync(io, CAMERA_HOST, CAMERA_PORT);

        auto cap = init_camera();
        int frame_id = 0;

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "フレームが空です。" << std::endl;
                continue;
            }
            cv::resize(frame, frame, cv::Size(IMAGE_W, IMAGE_H));

            // キー入力
            int key = cv::waitKey(10);
            if (key == 'q') break;

            std::vector<float> input = convert_to_chw(frame);
            std::vector<float> encoded = encoder.run(input);
            std::cout << "Encoded size: " << encoded.size()*4 << " byte" << std::endl;
            std::cout << "Output size: " << encoded.size() << " byte" << std::endl;
            std::vector<std::vector<float>> chunks = chunker::chunk_by_pixels<float>(encoded, CHUNK_C, CHUNK_H, CHUNK_W, CHUNK_PIXEL);
            debug_utils::print_chunk_info(chunks, CHUNK_H, CHUNK_W);
            std::cout << "Total Chunk: " << chunks.size() << std::endl;
            send_chunks(io, sender, frame_id, chunks);
            frame_id++;
        }
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return -1;
    }
}