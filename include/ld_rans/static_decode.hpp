#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <cassert>
#include "ryg_rans/rans_byte.h"

namespace static_rans {
// デコーダ側：静的モデル用
inline std::vector<uint8_t> decode(const std::vector<uint8_t>& compressed, size_t original_size) {
    std::array<RansDecSymbol, SYMBOL_COUNT> symbols;
    for (int i = 0; i < SYMBOL_COUNT; ++i) {
        RansDecSymbolInit(&symbols[i], i, 1);  // freq=1, cum=i
    }

    uint8_t* ptr = const_cast<uint8_t*>(compressed.data());
    RansState rans;
    RansDecInit(&rans, &ptr);

    std::vector<uint8_t> output(original_size);
    for (size_t i = 0; i < original_size; ++i) {
        uint32_t sym = rans & (PROB_SCALE - 1);
        uint8_t decoded = static_cast<uint8_t>(sym);
        output[i] = decoded;
        RansDecAdvanceSymbol(&rans, &ptr, &symbols[sym], PROB_BITS);
    }

    return output;
}
}
