// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/BaseBucketSummator.h"
#include "../SRPlatform/Interface/SRUtils.h"
#include "../SRPlatform/Interface/SRCast.h"

namespace SRPlat {

template<typename taNumber> class SRBucketSummatorSeq : public BaseBucketSummator<taNumber> {
  inline static __m128i __vectorcall OffsetsFrom4Exps(const __m128i exps32);
public:
  constexpr static inline size_t GetMemoryRequirementBytes();
  explicit inline SRBucketSummatorSeq(void* pMem);
  void inline ZeroBuckets();

  //// Passing |np| by value is only efficient so long as it fits and AVX register. There can be a separate function
  ////   without __vectorcall to take it by const reference.
  inline void __vectorcall CalcAdd(const SRNumPack<taNumber> np);
  inline void __vectorcall Add(const SRNumPack<taNumber> np, const __m256i exps);

  // Add a vector, in which only first nValid components are valid (e.g. the rest is out of range).
  inline void __vectorcall CalcAdd(const SRNumPack<taNumber> np, const SRVectCompCount nValid);

  inline taNumber ComputeSum();
};

template<typename taNumber> constexpr inline size_t SRBucketSummatorSeq<taNumber>::GetMemoryRequirementBytes() {
  return WorkerRowLengthBytes();
}

template<typename taNumber> inline SRBucketSummatorSeq<taNumber>::SRBucketSummatorSeq(void* pMem)
  : BaseBucketSummator(pMem) { }

template<typename taNumber> inline __m128i __vectorcall SRBucketSummatorSeq<taNumber>::OffsetsFrom4Exps(
  const __m128i exps32)
{
  const __m128i scaled = _mm_mul_epi32(exps32, taNumber::_cSizeBytes128_32);
  return scaled;
}

///////////////////////////////// SRBucketSummatorSeq<SRDoubleNumber> implementation ///////////////////////////////////

template<> inline void SRBucketSummatorSeq<SRDoubleNumber>::ZeroBuckets() {
  SRUtils::FillZeroVects<true>(SRCast::Ptr<__m256i>(_pBuckets), WorkerRowLengthBytes() >> SRSimd::_cLogNBytes);
}

template<> inline void __vectorcall SRBucketSummatorSeq<SRDoubleNumber>::CalcAdd(const SRNumPack<SRDoubleNumber> np) {
  const __m128i offsets = OffsetsFrom4Exps(SRSimd::ExtractExponents32<false>(np._comps));
  AddInternal4(np, offsets);
}

template<> inline void __vectorcall SRBucketSummatorSeq<SRDoubleNumber>::Add(const SRNumPack<SRDoubleNumber> np,
  const __m256i exps64)
{
  const __m128i offsets = OffsetsFrom4Exps(SRSimd::ExtractEven(exps64));
  AddInternal4(np, offsets);
}

template<> inline void __vectorcall SRBucketSummatorSeq<SRDoubleNumber>::CalcAdd(const SRNumPack<SRDoubleNumber> np,
  const SRVectCompCount nValid)
{
  const __m128i offsets = OffsetsFrom4Exps(SRSimd::ExtractExponents32<false>(np._comps));
  AddInternal4(np, nValid, offsets);
}

template<> inline SRDoubleNumber SRBucketSummatorSeq<SRDoubleNumber>::ComputeSum() {
  return SRDoubleNumber(SRSimd::StableSum<false>(SRCast::CPtr<double>(_pBuckets), BucketCount()));
}

} // namespace SRPlat
