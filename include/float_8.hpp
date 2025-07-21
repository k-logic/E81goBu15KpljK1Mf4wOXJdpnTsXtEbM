#pragma once
#include <stdint.h>
#include <math.h>

#define EXPONENT_BITS 4
#define EXPONENT_BIAS 7

struct float8_t {
    uint8_t fraction : 7 - EXPONENT_BITS;
    uint8_t exponent : EXPONENT_BITS;
    uint8_t sign : 1;
};

inline unsigned char float8_to_uchar(float8_t f8) {
    return (f8.sign << 7) | (f8.exponent << 3) | f8.fraction;
}

inline float8_t uchar_to_float8(unsigned char c) {
    float8_t f8;
    f8.sign     = (c >> 7) & 0x1;
    f8.exponent = (c >> 3) & 0xF;
    f8.fraction = c & 0x7;
    return f8;
}

inline float8_t increse_by_one(float8_t f8) {
    unsigned char temp = float8_to_uchar(f8);
    temp++;
    return uchar_to_float8(temp);
}

inline float8_t float_to_float8(float f) {
    float8_t result;
    uint32_t *p = (uint32_t *)&f;

    // Extract sign, exponent and fraction from the float
    uint32_t sign = (*p >> 31) & 0x1;
    uint32_t exponent = (*p >> 23) & 0xFF;
    uint32_t fraction = *p & 0x7FFFFF;

    // Save the sign
    result.sign = sign;

    if (exponent == 0x0) {          // Zero case (float32 subnormal are zero in float8)
        result.exponent = 0;        // 0000
        result.fraction = 0;        // 000
    } else if (exponent == 0xFF) {  // Infinity or NaN case
        result.exponent = 0xF;      // 1111
        result.fraction = 0;        // 000 - Only Infinity is represented
    } else {
        int32_t exp_value = exponent - 127;
        if (exp_value >= 8) {  // Infinity case
            result.exponent = 0xF;
            result.fraction = 0;
        } else if (exp_value <= -11) {  // Zero case
            result.exponent = 0;
            result.fraction = 0;
        } else if (exp_value == -10) {  // Subnormal case
            result.exponent = 0;
            result.fraction = 0;
            // (Rounding) Increase by one
            if (((fraction >> 22) & 0x01) == 1) {
                result = increse_by_one(result);
            }
        } else if (exp_value == -9) {  // Subnormal case
            result.exponent = 0;
            result.fraction = 0b001;
            // (Rounding) Increase by one
            if (((fraction >> 22) & 0x01) == 1) {
                result = increse_by_one(result);
            }
        } else if (exp_value == -8) {  // Subnormal case
            result.exponent = 0;
            result.fraction = 0b010 | ((fraction >> 22) & 0x01);
            // (Rounding) Increase by one
            if (((fraction >> 21) & 0x01) == 1) {
                result = increse_by_one(result);
            }
        } else if (exp_value == -7) {  // Subnormal case
            result.exponent = 0;
            result.fraction = 0b100 | ((fraction >> 21) & 0x03);
            // (Rounding) Increase by one
            if (((fraction >> 20) & 0x01) == 1) {
                result = increse_by_one(result);
            }
        } else {  // Normal case
            result.exponent = exp_value + EXPONENT_BIAS;
            result.fraction = fraction >> 20;  // Only keep 3 bits of fraction
            // (Rounding) Increase by one
            if (((fraction >> 19) & 0x01) == 1) {
                result = increse_by_one(result);
            }
        }
    }

    return result;
}

inline float float8_to_float(float8_t f8) {
    uint32_t sign = f8.sign;
    uint32_t exponent = 0;
    uint32_t fraction = 0;

    if (f8.exponent == 0x0) {    // Zero or subnormal case
        if (f8.fraction == 0) {  // Zero case
            exponent = 0;
            fraction = 0;
        } else if (f8.fraction >> 2 == 1) {  // Subnormal case
            exponent = 120;
            fraction = (f8.fraction & 0x03) << 21;
        } else if (f8.fraction >> 1 == 1) {  // Subnormal case
            exponent = 119;
            fraction = (f8.fraction & 0x01) << 22;
        } else if (f8.fraction == 1) {  // Subnormal case
            exponent = 118;
            fraction = 0;
        }
    } else if (f8.exponent == 0xF) {  // Infinity case - No NaN implemented for float8
        exponent = 0xFF;              // 1111 1111
        fraction = 0;
    } else {  // Normal case
        exponent = (f8.exponent - EXPONENT_BIAS + 127);
        fraction = f8.fraction << 20;
    }

    // Concatenate the sign, exponent and fraction to get the float
    uint32_t result = (sign << 31) | (exponent << 23) | fraction;

    // Cast to float to return the correct data type
    return *(float *)&result;
}
