// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {

class SRPLATFORM_API SRFastRandom {
public: // constants
  static const __m128i _cRShift;
private: // variables
  __m128i _s;
private: // methods
  void InitWithRD(const uint8_t since) {
    std::random_device rd;
    for (uint8_t i = since; i <= 3; i++) {
      _s.m128i_u32[i] = rd();
    }
  }
public: // methods
  explicit SRFastRandom() {
    for (uint8_t i = 0; i <= 1; i++) {
      if (!_rdrand64_step(_s.m128i_u64 + i)) {
        // printf("Oops: hardware random didn't succeed.\n");
        InitWithRD(i<<1);
        break;
      }
    }
  }
  explicit SRFastRandom(__m128i seed) : _s(seed) {
  }
  //TODO: instead, generate 4 random numbers at once with AVX2
  uint64_t Generate() {
    uint64_t x = _s.m128i_u64[0];
    const uint64_t y = _s.m128i_u64[1];
    _s.m128i_u64[0] = y;
    x ^= x << 23; // a
    const uint64_t t = _s.m128i_u64[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
    return t + y;
  }

  // Vectorized version seems much slower than scalar
  uint64_t SimdGenerate() {
    __m128i xy = _s; // x at xy[0], y at xy[1]
    _s.m128i_u64[0] = xy.m128i_u64[1];
    xy.m128i_u64[0] ^= xy.m128i_u64[0] << 23;
    const __m128i xyShifted = _mm_srlv_epi64(xy, _cRShift);
    const __m128i xorred = _mm_xor_si128(xy, xyShifted);
    _s.m128i_u64[1] = xorred.m128i_u64[0] ^ xorred.m128i_u64[1];
    return _s.m128i_u64[1] + xy.m128i_u64[1];
  }
};

} // namespace SRPlat
