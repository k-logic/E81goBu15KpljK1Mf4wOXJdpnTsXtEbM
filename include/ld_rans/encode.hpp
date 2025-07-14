#pragma once
#include <ld_rans/common.hpp>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <sys/mman.h>
#include <stdexcept>
#include <memory>
#include <ryg_rans/rans_byte.h>
#include <iostream>
#include <format>

namespace rans {
inline std::vector<uint8_t> encode(const std::vector<uint8_t>& input) {
    //std::cout << std::format("[rans::encode] PROB_SCALE = {}\n", PROB_SCALE);
    // 頻度マップ作成
    std::unordered_map<uint8_t, uint32_t> freq_map;
    for (uint8_t b : input) freq_map[b]++;
    std::vector<std::pair<uint8_t, uint32_t>> freq(freq_map.begin(), freq_map.end());
    std::sort(freq.begin(), freq.end());

    // モデル構築
    std::unordered_map<uint8_t, RansEncSymbol> symbols;
    uint32_t cum = 0;
    for (auto& [symbol, f] : freq) {
        RansEncSymbol sym;
        RansEncSymbolInit(&sym, cum, f, PROB_BITS);
        symbols[symbol] = sym;
        cum += f;
    }
    assert(cum <= PROB_SCALE);

    // mmap によるエンコードバッファ確保
    char* raw_buf = (char*)mmap(nullptr, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw_buf == MAP_FAILED) throw std::runtime_error("mmap failed");

    auto buf_sptr = std::shared_ptr<uint8_t>((uint8_t*)raw_buf, [=](uint8_t* p) { munmap(p, BUF_SIZE); });
    uint8_t* ptr = buf_sptr.get() + BUF_SIZE;

    // rANS エンコード
    RansState rans;
    RansEncInit(&rans);
    for (auto it = input.rbegin(); it != input.rend(); ++it) {
        RansEncPutSymbol(&rans, &ptr, &symbols[*it]);
    }
    RansEncFlush(&rans, &ptr);
    std::vector<uint8_t> compressed(ptr, buf_sptr.get() + BUF_SIZE);

    // freqヘッダーサイズ確認
    size_t freq_bytes = freq.size() * 3;
    size_t header_size = 2 + 1 + freq_bytes;  // payload_len, symbol_count, freq_bytes

    // 判定：圧縮サイズ合計 > 元データ長
    if (compressed.size() + header_size >= input.size()) {
        std::vector<uint8_t> fallback;
        uint16_t len = static_cast<uint16_t>(input.size());
        fallback.push_back(len & 0xFF);
        fallback.push_back((len >> 8) & 0xFF);
        fallback.insert(fallback.end(), input.begin(), input.end());
        return fallback;
    }
    
    std::vector<uint8_t> output;

    // payload_len（元データサイズ、2バイト）
    uint16_t payload_len = static_cast<uint16_t>(input.size());
    output.push_back(payload_len & 0xFF);
    output.push_back((payload_len >> 8) & 0xFF);

    // symbol_count（1バイト）
    uint8_t symbol_count = static_cast<uint8_t>(freq.size());
    output.push_back(symbol_count);

    // freq（3バイト × symbol_count）
    for (const auto& [symbol, f] : freq) {
        output.push_back(symbol);
        output.push_back(f & 0xFF);
        output.push_back((f >> 8) & 0xFF);
    }
    //std::cout << std::format("compressed.size() = {}\nfreq.size() = {}, total freq bytes = {}\n", compressed.size(), freq.size(), freq_bytes);
    
    // 圧縮データを後ろに追加
    output.insert(output.end(), compressed.begin(), compressed.end());

    return output;
}
}
