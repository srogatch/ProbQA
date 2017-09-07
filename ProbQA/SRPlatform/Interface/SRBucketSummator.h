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
  static inline constexpr size_t GetBucketCount();
  inline taNumber& ModBucket(const SRThreadPool::TThreadCount iWorker, const size_t iBucket);

public: // methods
  static inline size_t GetMemoryRequirementBytes(const SRThreadPool::TThreadCount nWorkers);
  explicit inline SRBucketSummator(const SRThreadPool::TThreadCount nWorkers, void* pMem);

  // Let each worker zero its buckets, so that they lay into L1/L2 cache of this worker.
  void inline ZeroBuckets(const  SRThreadPool::TThreadCount iWorker);

  //// Passing |np| by value is only efficient so long as it fits and AVX register. There can be a separate function
  ////   without __vectorcall to take it by const reference.
  void __vectorcall CalcAdd(const SRThreadPool::TThreadCount iWorker, SRNumPack<taNumber> np);
  void __vectorcall CalcAdd(const SRThreadPool::TThreadCount iWorker, SRNumPack<taNumber> np,
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

template<typename taNumber> taNumber& SRBucketSummator<taNumber>::ModBucket(const SRThreadPool::TThreadCount iWorker,
  const size_t iBucket)
{
  return _pBuckets[iWorker * GetBucketCount() + iBucket];
}

/////////////////////////////////// SRBucketSummator<SRDoubleNumber> implementation ////////////////////////////////////
template<> inline constexpr size_t SRBucketSummator<SRDoubleNumber>::GetBucketCount() {
  return size_t(1) << 11; // 11 bits of exponent
}

template<> inline void SRBucketSummator<SRDoubleNumber>::ZeroBuckets(const  SRThreadPool::TThreadCount iWorker) {
  SRUtils::FillZeroVects<true>(reinterpret_cast<__m256i*>(_pBuckets + iWorker * GetBucketCount()),
    GetBucketCount() >> 2);
}

template<> inline void __vectorcall SRBucketSummator<SRDoubleNumber>::CalcAdd(const SRThreadPool::TThreadCount iWorker,
  SRNumPack<SRDoubleNumber> np)
{
  const __m128i exps = SRSimd::ExtractExponents32<false>(np._comps);
  const __m128i indices = _mm_add_epi32(exps, _mm_set1_epi32(iWorker * GetBucketCount()));

  //TODO: refactor to _mm256_i32gather_pd() for CPUs on which it's faster. The below is faster on Ryzen, but not Skylake
  const __m256d old = _mm256_set_pd(_pBuckets[indices.m128i_i32[3]].GetValue(),
    _pBuckets[indices.m128i_i32[2]].GetValue(), _pBuckets[indices.m128i_i32[1]].GetValue(),
    _pBuckets[indices.m128i_i32[0]].GetValue());

  const __m256d sums = _mm256_add_pd(old, np._comps);

  //TODO: refactor to a scatter instruction when AVX512 is supported.
  _pBuckets[indices.m128i_i32[3]].SetValue(sums.m256d_f64[3]);
  _pBuckets[indices.m128i_i32[2]].SetValue(sums.m256d_f64[2]);
  _pBuckets[indices.m128i_i32[1]].SetValue(sums.m256d_f64[1]);
  _pBuckets[indices.m128i_i32[0]].SetValue(sums.m256d_f64[0]);
}
//void __vectorcall CalcAdd(const SRThreadPool::TThreadCount iWorker, SRNumPack<taNumber> np,
//  const typename SRNumPack<taNumber>::TCompsCount nComps);

} // namespace SRPlat
