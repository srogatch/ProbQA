// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRMacros.h"
#include "../SRPlatform/Interface/SRAccumulator.h"

namespace SRTest {
  class SRAccumVectDbl256TestHelper;
}

namespace SRPlat {

// 256-bit vector of Kahan summators for double precision scalars.
class SRAccumVectDbl256 {
  friend class ::SRTest::SRAccumVectDbl256TestHelper;
  __m256d _sum;
  __m256d _corr;

public:
  explicit SRAccumVectDbl256() : _sum(_mm256_setzero_pd()), _corr(_mm256_setzero_pd()) {}
  void Reset() {
    _sum = _mm256_setzero_pd();
    _corr = _mm256_setzero_pd();
  }
  inline SRAccumVectDbl256& __vectorcall Add(const __m256d value);
  inline SRAccumVectDbl256& __vectorcall Add(SRVectCompCount at, const double value);
  //Note: this method is not at maximum precision.
  inline double __vectorcall GetFullSum() const;
  inline double __vectorcall PreciseSum() const;
  //TODO: sum all components in this and fellow separately in AVX2, return the sum of this, and set fellowSum to the
  //  sum of fellow.
  inline double __vectorcall PairSum(const SRAccumVectDbl256& fellow, double& fellowSum) const;
};

FLOAT_PRECISE_BEGIN
inline SRAccumVectDbl256& __vectorcall SRAccumVectDbl256::Add(const __m256d value) {
  const __m256d y = _mm256_sub_pd(value, _corr);
  const __m256d t = _mm256_add_pd(_sum, y);
  _corr = _mm256_sub_pd(_mm256_sub_pd(t, _sum), y);
  _sum = t;
  return *this;
}

inline SRAccumVectDbl256& __vectorcall SRAccumVectDbl256::Add(SRVectCompCount at, const double value) {
  const double y = value - _corr.m256d_f64[at];
  const double t = _sum.m256d_f64[at] + y;
  _corr.m256d_f64[at] = (t - _sum.m256d_f64[at]) - y;
  _sum.m256d_f64[at] = t;
  return *this;
}

inline double __vectorcall SRAccumVectDbl256::GetFullSum() const {
  const __m256d interleaved = _mm256_hadd_pd(_corr, _sum);
  const __m128d corrSum = _mm_add_pd(_mm256_extractf128_pd(interleaved, 1), _mm256_castpd256_pd128(interleaved));
  return corrSum.m128d_f64[1] - corrSum.m128d_f64[0];
}

inline double __vectorcall SRAccumVectDbl256::PreciseSum() const {
// This is too complex for me at the moment w.r.t. the priorities.
//  __m128d sseSum, sseCorr;
//  {
//#define EASY_SET(atVar) _mm_set_pd(_sum.m256d_f64[atVar], _corr.m256d_f64[atVar])
//    sseSum = EASY_SET(0);
//    __m128d y = EASY_SET(1);
//    __m128d t = _mm_add_pd(sseSum, y);
//    sseCorr = _mm_sub_pd(_mm_sub_pd(t, sseSum), y);
//    sseSum = t;
//    for (int i = 2; i <= 3; i++) {
//      y = _mm_sub_pd(EASY_SET(i), sseCorr);
//      t = _mm_add_pd(sseSum, y);
//      sseCorr = _mm_sub_pd(_mm_sub_pd(t, sseSum), y);
//      sseSum = t;
//    }
//#undef EASY_SET
//  }
//  double scalSum, scalCorr;
//  scalSum = sseCorr.m128d_f64[0];
//  double y = -sseCorr.m128d_f64[1];
  SRAccumulator<SRDoubleNumber> ans(SRDoubleNumber::FromDouble(_corr.m256d_f64[3]));
  for (int i = 2; i >= 0; i--) {
    ans.Add(SRDoubleNumber::FromDouble(_corr.m256d_f64[i]));
  }
  ans.Neg();
  for (int i = 3; i >= 0; i--) {
    ans.Add(SRDoubleNumber::FromDouble(_sum.m256d_f64[i]));
  }
  return ans.Get().GetValue();
}

inline double __vectorcall SRAccumVectDbl256::PairSum(const SRAccumVectDbl256& fellow, double& fellowSum) const {
  // This is too complex for me at the moment w.r.t. the priorities.
//  __m256d avxSum, avxCorr;
//  {
//#define EASY_SET(atVar) _mm256_set_pd(fellow._sum.m256d_f64[atVar], _sum.m256d_f64[atVar], \
//  fellow._corr.m256d_f64[atVar], _corr.m256d_f64[atVar])
//    avxSum = EASY_SET(0);
//    __m256d y = EASY_SET(1);
//    __m256d t = _mm256_add_pd(avxSum, y);
//    avxCorr = _mm256_sub_pd(_mm256_sub_pd(t, avxSum), y);
//    avxSum = t;
//    for (int i = 2; i <= 3; i++) {
//      y = _mm256_sub_pd(EASY_SET(i), avxCorr);
//      t = _mm256_add_pd(avxSum, y);
//      avxCorr = _mm256_sub_pd(_mm256_sub_pd(t, avxSum), y);
//      avxSum = t;
//    }
//#undef EASY_SET
//  }
//  __m128d sseSum, sseCorr;
//  sseSum = _mm256_castpd256_pd128(avxCorr);
  __m128d sum = _mm_set_pd(fellow._corr.m256d_f64[3], _corr.m256d_f64[3]);
  __m128d corr = _mm_setzero_pd();
  for (int i = 2; i >= 0; i--) {
    const __m128d y = _mm_sub_pd(_mm_set_pd(fellow._corr.m256d_f64[i], _corr.m256d_f64[i]), corr);
    const __m128d t = _mm_add_pd(sum, y);
    corr = _mm_sub_pd(_mm_sub_pd(t, sum), y);
    sum = t;
  }
  sum = _mm_xor_pd(sum, SRSimd::_cDoubleSign128);
  corr = _mm_xor_pd(corr, SRSimd::_cDoubleSign128);
  for (int i = 3; i >= 0; i--) {
    const __m128d y = _mm_sub_pd(_mm_set_pd(fellow._sum.m256d_f64[i], _sum.m256d_f64[i]), corr);
    const __m128d t = _mm_add_pd(sum, y);
    corr = _mm_sub_pd(_mm_sub_pd(t, sum), y);
    sum = t;
  }
  fellowSum = sum.m128d_f64[1] - corr.m128d_f64[1];
  return sum.m128d_f64[0] - corr.m128d_f64[0];
}
FLOAT_PRECISE_END

} // namespace SRPlat
