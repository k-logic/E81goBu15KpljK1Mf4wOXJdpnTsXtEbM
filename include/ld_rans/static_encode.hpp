#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <cassert>
#include "ryg_rans/rans_byte.h"

namespace rans_static {
// エンコーダ側：静的モデル用
inline std::vector<uint8_t> encode(const std::vector<uint8_t>& input) {
    std::array<RansEncSymbol, SYMBOL_COUNT> symbols;
    for (int i = 0; i < SYMBOL_COUNT; ++i) {
        RansEncSymbolInit(&symbols[i], i, 1, PROB_BITS);  // freq=1, cum=i
    }

    std::vector<uint8_t> buffer(8192);
    uint8_t* ptr = buffer.data() + buffer.size();

    RansState rans;
    RansEncInit(&rans);

    for (auto it = input.rbegin(); it != input.rend(); ++it) {
        RansEncPutSymbol(&rans, &ptr, &symbols[*it]);
    }
    RansEncFlush(&rans, &ptr);

    std::vector<uint8_t> output(ptr, buffer.data() + buffer.size());
    return output;
}
}
