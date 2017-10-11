// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRMacros.h"

namespace SRPlat {

// Vector of Kahan summators
class SRAccumVectDbl {
  __m256d _sum;
  __m256d _corr;

public:
  explicit SRAccumVectDbl() : _sum(_mm256_setzero_pd()), _corr(_mm256_setzero_pd()) {}
  void Reset() {
    _sum = _mm256_setzero_pd();
    _corr = _mm256_setzero_pd();
  }
  inline SRAccumVectDbl& __vectorcall Add(const __m256d value);
  inline double __vectorcall GetFullSum();
};

FLOAT_PRECISE_BEGIN
inline SRAccumVectDbl& __vectorcall SRAccumVectDbl::Add(const __m256d value) {
  const __m256d y = _mm256_sub_pd(value, _corr);
  const __m256d t = _mm256_add_pd(_sum, y);
  _corr = _mm256_sub_pd(_mm256_sub_pd(t, _sum), y);
  _sum = t;
  return *this;
}
inline double __vectorcall SRAccumVectDbl::GetFullSum() {
  const __m256d interleaved = _mm256_hadd_pd(_corr, _sum);
  const __m128d corrSum = _mm_add_pd(_mm256_extractf128_pd(interleaved, 1), _mm256_castpd256_pd128(interleaved));
  return corrSum.m128d_f64[0] + corrSum.m128d_f64[1];
}
FLOAT_PRECISE_END

} // namespace SRPlat
