#pragma once
#include <ld_rans/common.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cassert>
#include <ryg_rans/rans_byte.h>

namespace rans {
inline std::vector<uint8_t> decode(const std::vector<uint8_t>& compressed) {
    size_t pos = 0;

    // payload_len（2バイト）
    uint16_t payload_len = compressed[pos++] | (compressed[pos++] << 8);

    // symbol_count（1バイト）
    uint8_t symbol_count = compressed[pos++];

    // freq テーブルの復元
    std::unordered_map<uint8_t, RansDecSymbol> symbols;
    std::vector<uint8_t> lut(PROB_SCALE);
    uint32_t cum = 0;

    for (int i = 0; i < symbol_count; ++i) {
        uint8_t symbol = compressed[pos++];
        uint16_t freq = compressed[pos++] | (compressed[pos++] << 8);

        RansDecSymbol sym;
        RansDecSymbolInit(&sym, cum, freq);
        symbols[symbol] = sym;

        for (uint32_t j = cum; j < cum + freq; ++j) {
            lut[j] = symbol;
        }

        cum += freq;
    }

    assert(cum <= PROB_SCALE);

    // === rANSデコード開始 ===
    uint8_t* ptr = const_cast<uint8_t*>(compressed.data() + pos);
    RansState rans;
    RansDecInit(&rans, &ptr);

    std::vector<uint8_t> output;
    output.reserve(payload_len);
    for (size_t i = 0; i < payload_len; ++i) {
        uint32_t s = rans & (PROB_SCALE - 1);
        uint8_t symbol = lut[s];
        output.push_back(symbol);
        RansDecAdvanceSymbol(&rans, &ptr, &symbols[symbol], PROB_BITS);
    }

    return output;
}
}
