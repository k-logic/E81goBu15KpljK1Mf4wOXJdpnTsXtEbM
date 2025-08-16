#include <iostream>
#include <string>
#include <CLI/CLI.hpp>

// ================================
// エンコーダー処理
// ================================
int run_encoder(
    const std::string& dst_ip,
    int dst_port,
    int send_buffer,
    const std::string& camera,
    int width, int height, int fps,
    const std::string& encoder_path,
    int in_c, int in_w, int in_h,
    int out_c, int out_w, int out_h,
    int chunk_pixel, int chunk_h, int chunk_w
) {
    std::cout << "[Encoder] dst_ip=" << dst_ip
              << " dst_port=" << dst_port
              << " send_buffer=" << send_buffer << "\n";
    std::cout << "[Encoder] camera=" << camera
              << " " << width << "x" << height << " @" << fps << "fps\n";
    std::cout << "[Encoder] model=" << encoder_path << "\n";
    // 実際のエンコード処理をここに実装
    return 0;
}

// ================================
// デコーダー処理
// ================================
int run_decoder(
    const std::string& listen_ip,
    int listen_port,
    int recv_buffer,
    const std::string& decoder_path,
    int in_c, int in_w, int in_h,
    int out_c, int out_w, int out_h
) {
    std::cout << "[Decoder] listen_ip=" << listen_ip
              << " listen_port=" << listen_port
              << " recv_buffer=" << recv_buffer << "\n";
    std::cout << "[Decoder] model=" << decoder_path << "\n";
    // 実際のデコード処理をここに実装
    return 0;
}

// ================================
// main
// ================================
int main(int argc, char** argv) {
    CLI::App app{"NeuralCodec CLI - エンコード/デコード統合ツール"};

    // ---- encode サブコマンド ----
    std::string dst_ip;
    int dst_port;
    int send_buffer;
    std::string camera;
    int width, height, fps;
    std::string encoder_path;
    int in_c_enc, in_w_enc, in_h_enc;
    int out_c_enc, out_w_enc, out_h_enc;
    int chunk_pixel, chunk_h, chunk_w;

    auto encode = app.add_subcommand("encode", "エンコードモードで実行します");
    encode->add_option("--dst-ip", dst_ip, "送信先IP")->required();
    encode->add_option("--dst-port", dst_port, "送信先ポート")->required();
    encode->add_option("--send-buffer", send_buffer, "送信バッファサイズ（バイト）")->default_val(131072);
    encode->add_option("--camera", camera, "カメラデバイスパス")->required();
    encode->add_option("--width", width, "カメラ横幅(px)")->required();
    encode->add_option("--height", height, "カメラ高さ(px)")->required();
    encode->add_option("--fps", fps, "フレームレート")->default_val(30);
    encode->add_option("--encoder-path", encoder_path, "エンコーダーモデルパス")->required();
    encode->add_option("--in-c", in_c_enc, "入力チャンネル数")->required();
    encode->add_option("--in-w", in_w_enc, "入力幅")->required();
    encode->add_option("--in-h", in_h_enc, "入力高さ")->required();
    encode->add_option("--out-c", out_c_enc, "出力チャンネル数")->required();
    encode->add_option("--out-w", out_w_enc, "出力幅")->required();
    encode->add_option("--out-h", out_h_enc, "出力高さ")->required();
    encode->add_option("--chunk-pixel", chunk_pixel, "チャンク内ピクセル数")->required();
    encode->add_option("--chunk-h", chunk_h, "チャンク高さ(px)")->required();
    encode->add_option("--chunk-w", chunk_w, "チャンク幅(px)")->required();
    encode->callback([&](){
        run_encoder(dst_ip, dst_port, send_buffer, camera, width, height, fps,
                    encoder_path, in_c_enc, in_w_enc, in_h_enc,
                    out_c_enc, out_w_enc, out_h_enc,
                    chunk_pixel, chunk_h, chunk_w);
    });

    // ---- decode サブコマンド ----
    std::string listen_ip;
    int listen_port;
    int recv_buffer;
    std::string decoder_path;
    int in_c_dec, in_w_dec, in_h_dec;
    int out_c_dec, out_w_dec, out_h_dec;

    auto decode = app.add_subcommand("decode", "デコードモードで実行します");
    decode->add_option("--listen-ip", listen_ip, "受信待ちIP")->default_val("0.0.0.0");
    decode->add_option("--listen-port", listen_port, "受信ポート")->required();
    decode->add_option("--recv-buffer", recv_buffer, "受信バッファサイズ（バイト）")->default_val(131072);
    decode->add_option("--decoder-path", decoder_path, "デコーダーモデルパス")->required();
    decode->add_option("--in-c", in_c_dec, "入力チャンネル数")->required();
    decode->add_option("--in-w", in_w_dec, "入力幅")->required();
    decode->add_option("--in-h", in_h_dec, "入力高さ")->required();
    decode->add_option("--out-c", out_c_dec, "出力チャンネル数")->required();
    decode->add_option("--out-w", out_w_dec, "出力幅")->required();
    decode->add_option("--out-h", out_h_dec, "出力高さ")->required();
    decode->callback([&](){
        run_decoder(listen_ip, listen_port, recv_buffer,
                    decoder_path, in_c_dec, in_w_dec, in_h_dec,
                    out_c_dec, out_w_dec, out_h_dec);
    });

    CLI11_PARSE(app, argc, argv);
}
