// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRThreadPool.h"
#include "../SRPlatform/Interface/SRRealNumber.h"
#include "SRDoubleNumber.h"

namespace SRPlat {

template<typename taNumber> class SRPLATFORM_API SRBucketSummator {
  taNumber *_pBuckets;
  const SRThreadPool::TThreadCount _nWorkers;

private: // methods
  static inline constexpr uint32_t GetBucketCount();
  inline taNumber& ModBucket(const SRThreadPool::TThreadCount iWorker, const uint32_t iBucket);
  inline static __m128i Get4Indices(const SRThreadPool::TThreadCount iWorker, const __m256d nums);
  inline static SRSimd::Packed64 Get2Indices(const SRThreadPool::TThreadCount iWorker, const __m128d nums);

public: // methods
  static inline size_t GetMemoryRequirementBytes(const SRThreadPool::TThreadCount nWorkers);
  explicit inline SRBucketSummator(const SRThreadPool::TThreadCount nWorkers, void* pMem);

  // Let each worker zero its buckets, so that they lay into L1/L2 cache of this worker.
  void inline ZeroBuckets(const  SRThreadPool::TThreadCount iWorker);

  //// Passing |np| by value is only efficient so long as it fits and AVX register. There can be a separate function
  ////   without __vectorcall to take it by const reference.
  void __vectorcall CalcAdd(const SRThreadPool::TThreadCount iWorker, const SRNumPack<taNumber> np);
  void __vectorcall CalcAdd(const SRThreadPool::TThreadCount iWorker, const SRNumPack<taNumber> np,
    const typename SRNumPack<taNumber>::TCompsCount nComps);
};

template<typename taNumber> inline size_t SRBucketSummator<taNumber>::GetMemoryRequirementBytes(
  const SRThreadPool::TThreadCount nWorkers)
{
  return sizeof(taNumber) * GetBucketCount() * size_t(nWorkers);
}

template<typename taNumber> inline SRBucketSummator<taNumber>::SRBucketSummator(
  const SRThreadPool::TThreadCount nWorkers, void* pMem)
  : _nWorkers(nWorkers), _pBuckets(reinterpret_cast<taNumber*>(pMem))
{
  ZeroBuckets();
}

template<typename taNumber> inline taNumber& SRBucketSummator<taNumber>::ModBucket(
  const SRThreadPool::TThreadCount iWorker, const uint32_t iBucket)
{
  return _pBuckets[uint32_t(iWorker) * GetBucketCount() + iBucket];
}

template<typename taNumber> inline __m128i SRBucketSummator<taNumber>::Get4Indices(
  const SRThreadPool::TThreadCount iWorker, const __m256d nums)
{
  const __m128i exps = SRSimd::ExtractExponents32<false>(nums);
  const __m128i indices = _mm_add_epi32(exps, _mm_set1_epi32(iWorker * GetBucketCount()));
  return indices;
}

template<typename taNumber> inline SRSimd::Packed64 SRBucketSummator<taNumber>::Get2Indices(
  const SRThreadPool::TThreadCount iWorker, const __m128d nums)
{
  const SRSimd::Packed64 exps = SRSimd::ExtractExponents32<false>(nums);
  constexpr uint64_t packedBCs = (uint64_t(GetBucketCount()) << 32) + GetBucketCount();
  const SRSimd::Packed64 indices(packedBCs * iWorker + exps._u64);
  return indices;
}

/////////////////////////////////// SRBucketSummator<SRDoubleNumber> implementation ////////////////////////////////////
template<> inline constexpr uint32_t SRBucketSummator<SRDoubleNumber>::GetBucketCount() {
  return uint32_t(1) << 11; // 11 bits of exponent
}

template<> inline void SRBucketSummator<SRDoubleNumber>::ZeroBuckets(const  SRThreadPool::TThreadCount iWorker) {
  SRUtils::FillZeroVects<true>(reinterpret_cast<__m256i*>(_pBuckets + iWorker * GetBucketCount()),
    GetBucketCount() >> 2);
}

template<> inline void __vectorcall SRBucketSummator<SRDoubleNumber>::CalcAdd(const SRThreadPool::TThreadCount iWorker,
  const SRNumPack<SRDoubleNumber> np)
{
  const __m128i indices = Get4Indices(iWorker, np._comps);

  //TODO: refactor to _mm256_i32gather_pd() for CPUs on which it's faster. The below is faster on Ryzen, but not Skylake
  const __m256d old = _mm256_set_pd(_pBuckets[indices.m128i_u32[3]].GetValue(),
    _pBuckets[indices.m128i_u32[2]].GetValue(), _pBuckets[indices.m128i_u32[1]].GetValue(),
    _pBuckets[indices.m128i_u32[0]].GetValue());

  const __m256d sums = _mm256_add_pd(old, np._comps);

  //TODO: refactor to a scatter instruction when AVX512 is supported.
  _pBuckets[indices.m128i_u32[3]].SetValue(sums.m256d_f64[3]);
  _pBuckets[indices.m128i_u32[2]].SetValue(sums.m256d_f64[2]);
  _pBuckets[indices.m128i_u32[1]].SetValue(sums.m256d_f64[1]);
  _pBuckets[indices.m128i_u32[0]].SetValue(sums.m256d_f64[0]);
}

template<> inline void __vectorcall SRBucketSummator<SRDoubleNumber>::CalcAdd(const SRThreadPool::TThreadCount iWorker,
  const SRNumPack<SRDoubleNumber> np, const typename SRNumPack<SRDoubleNumber>::TCompsCount nComps)
{
  assert(nComps <= 4);
  switch (nComps) {
  case 0:
    return;
  case 4:
    return CalcAdd(iWorker, np); // full vector addition
  case 1: {
    const int16_t exponent = SRNumTraits<double>::ExtractExponent<false>(np._comps.m256d_f64[0]);
    ModBucket(iWorker, exponent) += np._comps.m256d_f64[0];
    return;
  }

  case 3: {
    const int16_t exponent = SRNumTraits<double>::ExtractExponent<false>(np._comps.m256d_f64[2]);
    ModBucket(iWorker, exponent) += np._comps.m256d_f64[2];
  }
  // Fall through
  case 2: {
    SRSimd::Packed64 indices = Get2Indices(iWorker, _mm256_castpd256_pd128(np._comps));
    const __m128d old = _mm_set_pd(_pBuckets[indices._u32[1]].GetValue(), _pBuckets[indices._u32[0]].GetValue());
    const __m128d sums = _mm_add_pd(old, _mm256_castpd256_pd128(np._comps));
    _pBuckets[indices._u32[1]].SetValue(sums.m128d_f64[1]);
    _pBuckets[indices._u32[0]].SetValue(sums.m128d_f64[0]);
    break;
  }
  default:
    __assume(0);
  }

  //const __m128i indices = Get4Indices(iWorker, np._comps);
  //uint8_t lastComp;
  //if (nComps >= 2) {
  //  lastComp = 2;
  //  const __m128d old = _mm_set_pd(_pBuckets[indices.m128i_u32[1]].GetValue(),
  //    _pBuckets[indices.m128i_u32[0]].GetValue());
  //  const __m128d sums = _mm_add_pd(old, _mm256_castpd256_pd128(np._comps));
  //  _pBuckets[indices.m128i_u32[1]].SetValue(sums.m128d_f64[1]);
  //  _pBuckets[indices.m128i_u32[0]].SetValue(sums.m128d_f64[0]);
  //}
  //else {
  //  lastComp = 0;
  //}
  //if (lastComp < nComps) {

  //}
  //// At expense of adding values separately (rather than a SIMD add), here we save extra loads from memory of buckets
  ////   not used.
  //for (uint8_t i = 0; i < nComps; i++) {
  //  _pBuckets[indices.m128i_u32[i]].ModValue() += np._comps.m256d_f64[i];
  //}
}

} // namespace SRPlat
