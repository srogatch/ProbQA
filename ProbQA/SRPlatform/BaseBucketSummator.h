// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRDoubleNumber.h"

namespace SRPlat {

template<typename taNumber> class BaseBucketSummator {
protected:
  taNumber *_pBuckets;

protected: // methods
  explicit BaseBucketSummator(void *pMem);
  static inline constexpr uint32_t BucketCount();
  static inline constexpr int32_t WorkerRowLengthBytes();
  ATTR_NOALIAS inline taNumber& ModOffs(const size_t byteOffs);
  inline void AddInternal4(const SRNumPack<taNumber> np, const __m128i offsets);

public: // methods
};

template<typename taNumber> inline BaseBucketSummator<taNumber>::BaseBucketSummator(void *pMem)
  : _pBuckets(static_cast<taNumber*>(pMem)) { }

template<typename taNumber> inline constexpr int32_t BaseBucketSummator<taNumber>::WorkerRowLengthBytes() {
  // If (sizeof(taNumber)*GetBucketCount()) is not a multiple of SIMD size, we should add padding so to keep each
  //   worker's piece of the array aligned for SIMD.
  return (BucketCount() * sizeof(taNumber) + SRSimd::_cByteMask) & (~SRSimd::_cByteMask);
}

template<typename taNumber> ATTR_NOALIAS inline taNumber& BaseBucketSummator<taNumber>::ModOffs(const size_t byteOffs) {
  return *SRCast::Ptr<taNumber>(SRCast::Ptr<uint8_t>(_pBuckets) + byteOffs);
}

///////////////////////////////// BaseBucketSummator<SRDoubleNumber> implementation ////////////////////////////////////
template<> inline constexpr uint32_t BaseBucketSummator<SRDoubleNumber>::BucketCount() {
  return uint32_t(1) << 11; // 11 bits of exponent
}

template<> inline void BaseBucketSummator<SRDoubleNumber>::AddInternal4(const SRNumPack<SRDoubleNumber> np,
  const __m128i offsets)
{
  ModOffs(offsets.m128i_u32[0]) += np._comps.m256d_f64[0];
  ModOffs(offsets.m128i_u32[1]) += np._comps.m256d_f64[1];
  ModOffs(offsets.m128i_u32[2]) += np._comps.m256d_f64[2];
  ModOffs(offsets.m128i_u32[3]) += np._comps.m256d_f64[3];

  //TODO: this requires fast conflict detection
  //SRDoubleNumber &b0 = ModOffs(offsets.m128i_u32[0]);
  //SRDoubleNumber &b1 = ModOffs(offsets.m128i_u32[1]);
  //SRDoubleNumber &b2 = ModOffs(offsets.m128i_u32[2]);
  //SRDoubleNumber &b3 = ModOffs(offsets.m128i_u32[3]);

  //const __m256d old = _mm256_set_pd(b3.GetValue(), b2.GetValue(), b1.GetValue(), b0.GetValue());
  //const __m256d sums = _mm256_add_pd(old, np._comps);

  //b3.SetValue(sums.m256d_f64[3]);
  //b2.SetValue(sums.m256d_f64[2]);
  //b1.SetValue(sums.m256d_f64[1]);
  //b0.SetValue(sums.m256d_f64[0]);
}

} // namespace SRPlat
