// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {

class SRPLATFORM_API SRFastRandom {
public: // constants
  static const uint8_t _cnAtOnce = sizeof(__m256i) / sizeof(uint64_t);

private: // variables
  __m256i _s[2];
  __m256i _preGen;
  uint8_t _iNextGen = 0;

private: // methods
  inline void InitWithRD(const uint8_t since) {
    std::random_device rd;
    for (uint8_t i = since; i < sizeof(_s) / sizeof(uint32_t); i++) {
      _s[i >> 3].m256i_u32[i & 7] = rd();
    }
  }

public: // methods
  static SRFastRandom& ThreadLocal();

  explicit SRFastRandom() {
    for (uint8_t i = 0; i < sizeof(_s) / sizeof(uint64_t); i++) {
      if (!_rdrand64_step(_s[i>>2].m256i_u64 + (i&3))) {
        // Can't log this because this likely happens because of too many hardware randoms per second.
        // printf("Oops: hardware random didn't succeed.\n");
        InitWithRD(i<<1);
        break;
      }
    }
  }
  explicit SRFastRandom(const __m256i s0, const __m256i s1) : _s{s0, s1} {
  }

  // Use entropy adapters for generating random values smaller that 64 bits.
  template<typename taResult> taResult Generate();

  template<> uint64_t Generate() {
    if (_iNextGen == 0) {
      const __m256i rn = Generate<__m256i>();
      _mm256_store_si256(&_preGen, rn);
      _iNextGen = 1;
      return rn.m256i_u64[0];
    }
    const uint64_t answer = _preGen.m256i_u64[_iNextGen];
    _iNextGen = (_iNextGen+1) & 3;
    return answer;
  }

  // Generate 4 random numbers at once with AVX2
  template<> __m256i __vectorcall Generate() {
    __m256i x = _mm256_load_si256(_s+0);
    const __m256i y = _mm256_load_si256(_s+1);
    _mm256_store_si256(_s + 0, y);
    x = _mm256_xor_si256(x, _mm256_slli_epi64(x, 23));
    const __m256i yrs26 = _mm256_srli_epi64(y, 26);
    const __m256i xrs17 = _mm256_srli_epi64(x, 17);
    const __m256i x_xor_y = _mm256_xor_si256(x, y);
    const __m256i xrs17_xor_yrs26 = _mm256_xor_si256(xrs17, yrs26);
    const __m256i t = _mm256_xor_si256(x_xor_y, xrs17_xor_yrs26);
    _mm256_store_si256(_s + 1, t);
    return _mm256_add_epi64(t, y);
  }
};

class SRPLATFORM_API SREntropyAdapter {
  SRFastRandom& _fr;
  uint64_t _remaining = 0;
public:
  explicit SREntropyAdapter(SRFastRandom& fr) : _fr(fr) {
  }
  // Supports unsigned types. Signed may give a warning.
  template<typename T> T Generate(const T lim) {
    assert(lim > 0);
    if (_remaining < uint64_t(lim-1)) {
      _remaining = _fr.Generate<uint64_t>();
    }
    const T answer = _remaining % lim;
    _remaining /= lim;
    return answer;
  }
};

} // namespace SRPlat
