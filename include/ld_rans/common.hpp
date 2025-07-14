#pragma once

namespace rans {
    constexpr int PROB_BITS = 12;
    constexpr int PROB_SCALE = 1 << PROB_BITS;
    constexpr int SYMBOL_COUNT = 256;
    constexpr size_t BUF_SIZE = 8092;
}