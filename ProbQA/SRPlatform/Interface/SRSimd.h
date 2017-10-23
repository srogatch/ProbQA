// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRNumTraits.h"
#include "../SRPlatform/Interface/SRPacked64.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"

namespace SRPlat {

class SRPLATFORM_API SRSimd {
  template<typename taResult, typename taParam> struct CastImpl;

public:
  typedef __m256i TIntSimd;

  static constexpr uint8_t _cLogNBits = 8; //AVX2, 256 bits, log2(256)=8
  static constexpr uint8_t _cLogNBytes = _cLogNBits - 3;
  static constexpr uint8_t _cLogNComps64 = _cLogNBytes - 3;
  static constexpr SRVectCompCount _cNComps64 = (1 << _cLogNComps64);
  static constexpr size_t _cNBits = 1 << _cLogNBits;
  static constexpr size_t _cNBytes = 1 << _cLogNBytes;

  static constexpr size_t _cByteMask = _cNBytes - 1;
  static constexpr size_t _cBitMask = _cNBits - 1;
  
  //// SIMD constants follow. For improved cache usage they must be adjacent.
  static const __m256i _cDoubleExpMaskUp;
  static const __m256i _cDoubleExp0Up;
  static const __m256i _cDoubleExp0Down;
  static const __m128i _cDoubleExpMaskDown32;
  static const __m128i _cDoubleExp0Down32;
  static const __m128d _cDoubleSign128;
private:
  static const __m256i _cSet1MsbOffs;
  static const __m256i _cSet1LsbOffs;
  static constexpr uint8_t _cnStbqEntries = 1 << 4;
  static const uint32_t _cStbqTable[_cnStbqEntries];

public:
  static size_t VectsFromBytes(const size_t nBytes) {
    return SRMath::RShiftRoundUp(nBytes, _cLogNBytes);
  }
  static size_t VectsFromBits(const uint64_t nBits) {
    return SRMath::RShiftRoundUp(nBits, _cLogNBits);
  }

  template<typename taComp> static size_t VectsFromComps(const size_t nComps) {
    static_assert(sizeof(taComp) <= _cNBytes, "Current limitation: component must be no larger than SIMD vector.");
    constexpr size_t logCompBytes = SRMath::StaticCeilLog2(sizeof(taComp));
    // SRMath::StaticIsPowOf2() would also work here, but we want to check correctness of logCompBytes too.
    static_assert(sizeof(taComp) == (1 << logCompBytes), "Component size must be exactly a power of 2.");
    return SRMath::RShiftRoundUp(nComps, _cLogNBytes - logCompBytes);
  }

  template<typename taResult, typename taParam> static  taResult __vectorcall Cast(const taParam par);

  template<bool taCache, typename taVect> static std::enable_if_t<sizeof(taVect)==sizeof(__m256i), taVect> __vectorcall
  Load(const taVect *const p)
  {
    const __m256i *const genP = SRCast::CPtr<__m256i>(p);
    //TODO: verify that taCache based branchings are compile-time
    const __m256i ans = (taCache ? _mm256_load_si256(genP) : _mm256_stream_load_si256(genP));
    return Cast<taVect>(ans);
  }

  template<bool taCache, typename taVect> static std::enable_if_t<sizeof(taVect)==sizeof(__m256i)> __vectorcall
  Store(taVect *p, const taVect v)
  {
    __m256i *genP = SRCast::Ptr<__m256i>(p);
    //TODO: verify that this casting turns into no-op in assembly, so that the value just stays in the register
    const __m256i genV = Cast<__m256i>(v);
    taCache ? _mm256_store_si256(genP, genV) : _mm256_stream_si256(genP, genV);
  }

  constexpr static size_t GetPaddedBytes(const size_t nUnpaddedBytes) {
    return (nUnpaddedBytes + _cByteMask) & (~_cByteMask);
  }
  constexpr static void* AlignPtr(void *p, const size_t maxPadding) {
    const uintptr_t upOrig = reinterpret_cast<uintptr_t>(p);
    const uintptr_t upAligned = (upOrig + _cByteMask) & (~_cByteMask);
    if (upAligned - upOrig > maxPadding) {
      throw SRException(SRMessageBuilder("Insufficient max padding ")(maxPadding)(" for aligning ")(upOrig)
        .GetOwnedSRString());
    }
    return reinterpret_cast<void*>(upAligned);
  }

  template<size_t taItemSize> constexpr static size_t PaddedBytesFromItems(const size_t nItems) {
    return GetPaddedBytes(nItems * taItemSize);
  }

  // set Most Significant bits to 1
  ATTR_NOALIAS static __m256i __vectorcall SetHighBits1(const uint16_t nMsBits1) {
    const __m256i ones = _mm256_set1_epi8(-1i8);
    __m256i shift = _mm256_set1_epi32(nMsBits1);
    shift = _mm256_subs_epu16(_cSet1MsbOffs, shift);
    return _mm256_sllv_epi32(ones, shift);
  }

  // set Least Significant bits to 1
  ATTR_NOALIAS static __m256i __vectorcall SetLowBits1(const uint16_t nLsBits1) {
    const __m256i ones = _mm256_set1_epi8(-1i8);
    __m256i shift = _mm256_set1_epi32(nLsBits1);
    shift = _mm256_subs_epu16(_cSet1LsbOffs, shift);
    return _mm256_srlv_epi32(ones, shift);
  }

  ATTR_NOALIAS static __m256i __vectorcall BroadcastBytesToComps64(const uint32_t bytes) {
    const __m128i source = _mm_cvtsi32_si128(bytes);
    return _mm256_cvtepi8_epi64(source);
  }

  // set Most Significant 64-bit components to all-one bits
  ATTR_NOALIAS static __m256i __vectorcall SetHighComps64(const SRVectCompCount nComps) {
    const uint32_t packed = static_cast<uint32_t>((uint64_t(-1i32) >> (nComps << 3)) - 1);
    return BroadcastBytesToComps64(packed);
  }

  ATTR_NOALIAS static __m256i __vectorcall SetLowComps64(const SRVectCompCount nComps) {
    const uint32_t packed = static_cast<uint32_t>((1ui64 << (nComps << 3)) - 1);
    return BroadcastBytesToComps64(packed);
  }

  template<bool taNorm0> ATTR_NOALIAS static __m256i __vectorcall ExtractExponents64(const __m256d nums) {
    const __m256i exps = _mm256_srli_epi64(_mm256_and_si256(_cDoubleExpMaskUp, _mm256_castpd_si256(nums)),
      SRNumTraits<double>::_cExponentOffs);
    if (!taNorm0) {
      return exps;
    }
    return _mm256_sub_epi64(exps, _cDoubleExp0Down);
  }

  // Extract components 1,3,5,7
  ATTR_NOALIAS static __m128i __vectorcall ExtractOdd(const __m256i vect) {
    const __m128i hiLane = _mm256_extracti128_si256(vect, 1);
    const __m128i loLane = _mm256_castsi256_si128(vect);
    return _mm_castps_si128(_mm_shuffle_ps(
      _mm_castsi128_ps(loLane), _mm_castsi128_ps(hiLane), _MM_SHUFFLE(3, 1, 3, 1)));
  }

  // Extract components 1,3,5,7. This should be faster than the integer version due to less casts.
  ATTR_NOALIAS static __m128 __vectorcall ExtractOdd(const __m256 vect) {
    const __m128 hiLane = _mm256_extractf128_ps(vect, 1);
    const __m128 loLane = _mm256_castps256_ps128(vect);
    return _mm_shuffle_ps(loLane, hiLane, _MM_SHUFFLE(3, 1, 3, 1));
  }

  // Extract components 0,2,4,6
  ATTR_NOALIAS static __m128i __vectorcall ExtractEven(const __m256i vect) {
    const __m128i hiLane = _mm256_extracti128_si256(vect, 1);
    const __m128i loLane = _mm256_castsi256_si128(vect);
    return _mm_castps_si128(_mm_shuffle_ps(
      _mm_castsi128_ps(loLane), _mm_castsi128_ps(hiLane), _MM_SHUFFLE(2, 0, 2, 0)));
  }

  // Extract components 0,2,4,6. This should be faster than the integer version due to less casts.
  ATTR_NOALIAS static __m128 __vectorcall ExtractEven(const __m256 vect) {
    const __m128 hiLane = _mm256_extractf128_ps(vect, 1);
    const __m128 loLane = _mm256_castps256_ps128(vect);
    return _mm_shuffle_ps(loLane, hiLane, _MM_SHUFFLE(2, 0, 2, 0));
  }

  template<bool taNorm0> ATTR_NOALIAS static __m128i __vectorcall ExtractExponents32(const __m256d nums) {
    // Don't reuse ExtractOdd, so to avoid integer-float transition penalties.
    const __m128 hiLane = _mm_castpd_ps(_mm256_extractf128_pd(nums, 1));
    const __m128 loLane = _mm_castpd_ps(_mm256_castpd256_pd128(nums));
    const __m128i high32 = _mm_castps_si128(_mm_shuffle_ps(loLane, hiLane, _MM_SHUFFLE(3, 1, 3, 1)));
    const __m128i exps = _mm_and_si128(_cDoubleExpMaskDown32,
      _mm_srli_epi32(high32, SRNumTraits<double>::_cExponentOffs - 32));
    if constexpr (!taNorm0) {
      return exps;
    }
    return _mm_sub_epi32(exps, _cDoubleExp0Down32);
  }

  template<bool taNorm0> ATTR_NOALIAS static SRPacked64 __vectorcall ExtractExponents32(const __m128d nums) {
    const __m128i shuffled = _mm_shuffle_epi32(_mm_castpd_si128(nums), _MM_SHUFFLE(0, 0, 3, 1));
    const SRPacked64 high32 = SRPacked64::SetToComp<0>(shuffled);
    constexpr SRPacked64 expMask = SRPacked64::Set1U32(SRNumTraits<double>::_cExponentMaskDown);
    const SRPacked64 exps((high32._u64 >> (SRNumTraits<double>::_cExponentOffs - 32)) & expMask._u64);
    if (!taNorm0) {
      return exps;
    }
    constexpr SRPacked64 normalizer = SRPacked64::Set1U32(SRNumTraits<double>::_cExponent0Down);
    return SRPacked64(exps._u64 - normalizer._u64);
  }

  ATTR_NOALIAS static __m256d __vectorcall MakeExponent0(const __m256d nums) {
    const __m256d e0nums = _mm256_or_pd(_mm256_castsi256_pd(_cDoubleExp0Up),
      _mm256_andnot_pd(_mm256_castsi256_pd(_cDoubleExpMaskUp), nums));
    return e0nums;
  }

  ATTR_NOALIAS static __m256d __vectorcall ReplaceExponents(const __m256d nums, const __m256i exps) {
    const __m256d newExps = _mm256_castsi256_pd(_mm256_slli_epi64(exps, SRNumTraits<double>::_cExponentOffs));
    const __m256d newNums = _mm256_or_pd(newExps, _mm256_andnot_pd(_mm256_castsi256_pd(_cDoubleExpMaskUp), nums));
    return newNums;
  }

  ATTR_NOALIAS static __m256d __vectorcall HorizAddStraight(const __m256d a, const __m256d b) {
    const __m256d crossed = _mm256_hadd_pd(a, b);
    const __m256d straight = _mm256_permute4x64_pd(crossed, _MM_SHUFFLE(3, 1, 2, 0));
    return straight;
  }

  ATTR_NOALIAS static double __vectorcall FullHorizSum(const __m256d a) {
    const __m256d b = _mm256_permute4x64_pd(a, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128d laneSums = _mm_hadd_pd(_mm256_extractf128_pd(b, 1), _mm256_castpd256_pd128(b));
    return laneSums.m128d_f64[0] + laneSums.m128d_f64[1];
  }

  template<bool taCache> ATTR_NOALIAS inline static double StableSum(const double *PTR_RESTRICT p,
    const size_t nDoubles);

  ATTR_NOALIAS static __m256i __vectorcall MaxI64(const __m256i a, const __m256i b, const __m256i maskA) {
    const __m256i isAGreater = _mm256_cmpgt_epi64(a, b);
    const __m256i maxes = _mm256_blendv_epi8(b, a, _mm256_or_si256(isAGreater, maskA));
    return maxes;
  }

  ATTR_NOALIAS static __m256i __vectorcall MaxI64(const __m256i a, const __m256i b) {
    const __m256i isAGreater = _mm256_cmpgt_epi64(a, b);
    const __m256i maxes = _mm256_blendv_epi8(b, a, isAGreater);
    return maxes;
  }

  ATTR_NOALIAS static int64_t __vectorcall FullHorizMaxI64(const __m256i a) {
    const __m128i hiLane = _mm256_extracti128_si256(a, 1);
    const __m128i loLane = _mm256_castsi256_si128(a);
    const __m128i isLoGreater = _mm_cmpgt_epi64(loLane, hiLane);
    const __m128i maxes = _mm_blendv_epi8(hiLane, loLane, isLoGreater);
    return std::max(maxes.m128i_i64[0], maxes.m128i_i64[1]);
  }

  // Version for infrequent calls, so to minimize cache footprint.
  ATTR_NOALIAS static __m256i __vectorcall SetToBitQuadCold(const uint8_t bitQuad) {
    assert(bitQuad < _cnStbqEntries);
    return _mm256_set_epi64x(-int64_t(bitQuad >> 3), -int64_t((bitQuad >> 2) & 1), -int64_t((bitQuad >> 1) & 1),
      -int64_t(bitQuad & 1));
  }

  // Version for frequent calls in a bottleneck code.
  ATTR_NOALIAS static __m256i __vectorcall SetToBitQuadHot(const uint8_t bitQuad) {
    assert(bitQuad < _cnStbqEntries);
    return BroadcastBytesToComps64(_cStbqTable[bitQuad]);
  }

  // Checks 4 32-bit components for conflicts. If components from lower to higher are abcd, returns:
  //   Bit 0: a conflicts with b
  //   Bit 1: b conflicts with c
  //   Bit 2: c conflicts with d
  //   Bit 3: d conflicts with a
  //   Bit 4: a conflicts with c
  //   Bit 5: b conflicts with d
  // The other bits are cleared. So it returns 0 if no conflicts.
  ATTR_NOALIAS static uint8_t __vectorcall DetectConflicts32(const __m128i comps) {
    // Compare:
    //   abcdabcd
    //   bcdacdXX
    const __m256i repeated = _mm256_broadcastsi128_si256(comps);
    const __m256i shuffled = _mm256_set_m128i(_mm_set_epi64x(-1, _mm_extract_epi64(comps, 1)),
      _mm_shuffle_epi32(comps, _MM_SHUFFLE(0, 3, 2, 1)));
    const __m256i equal = _mm256_cmpeq_epi32(repeated, shuffled);
    int conflicts = _mm256_movemask_ps(_mm256_castsi256_ps(equal));
    // Clear bits 6 and 7, if occasionally c or d is equal to -1.
    return static_cast<uint8_t>(conflicts & 63);
  }

  template<typename cbFetch, typename cbProcess> ATTR_NOALIAS static void ForTailF64(
    const SRVectCompCount nComps, const cbFetch &PTR_RESTRICT fetch, const cbProcess &PTR_RESTRICT process,
    const double placeholder);

  template<typename cbFetch, typename cbProcess> ATTR_NOALIAS static void ForTailI64(
    const SRVectCompCount nComps, const cbFetch &PTR_RESTRICT fetch, const cbProcess &PTR_RESTRICT process,
    const int64_t placeholder);

};

template<typename cbFetch, typename cbProcess> ATTR_NOALIAS void SRSimd::ForTailF64(const SRVectCompCount nComps,
  const cbFetch &PTR_RESTRICT fetch, const cbProcess &PTR_RESTRICT process, const double placeholder)
{
  __m256d vect;
  switch (nComps) {
  case 0:
    return;
  case 1: {
    const __m128d hi = _mm_set1_pd(placeholder);
    const __m128d lo = _mm_set_pd(placeholder, fetch(0));
    vect = _mm256_set_m128d(hi, lo);
    break;
  }
  case 2: {
    const __m128d hi = _mm_set1_pd(placeholder);
    const __m128d lo = _mm_set_pd(fetch(1), fetch(0));
    vect = _mm256_set_m128d(hi, lo);
    break;
  }
  case 3: {
    vect = _mm256_set_pd(placeholder, fetch(2), fetch(1), fetch(0));
    break;
  }
  default:
    __assume(0);
  }
  process(vect);
}

template<typename cbFetch, typename cbProcess> ATTR_NOALIAS void SRSimd::ForTailI64(const SRVectCompCount nComps,
  const cbFetch &PTR_RESTRICT fetch, const cbProcess &PTR_RESTRICT process, const int64_t placeholder)
{
  __m256i vect;
  switch (nComps) {
  case 0:
    return;
  case 1: {
    vect = _mm256_insert_epi64(_mm256_set1_epi64x(placeholder), fetch(0), 0);
    break;
  }
  case 2: {
    const __m128i hi = _mm_set1_epi64x(placeholder);
    const __m128i lo = _mm_set_epi64x(fetch(1), fetch(0));
    vect = _mm256_set_m128i(hi, lo);
    break;
  }
  case 3: {
    vect = _mm256_set_epi64x(placeholder, fetch(2), fetch(1), fetch(0));
    break;
  }
  default:
    __assume(0);
  }
  process(vect);
}

FLOAT_PRECISE_BEGIN
template<bool taCache> ATTR_NOALIAS inline double SRSimd::StableSum(const double *PTR_RESTRICT p,
  const size_t nDoubles)
{
  const size_t nVects = nDoubles >> 2;
  const size_t iTail = nVects << 2;
  const SRVectCompCount nTail = static_cast<SRVectCompCount>(nDoubles - iTail);
  double tailSum;

  switch (nTail) {
  case 0:
    tailSum = 0;
    break;
  case 1:
    tailSum = p[iTail];
    break;
  case 2:
    tailSum = p[iTail] + p[iTail + 1];
    break;
  case 3:
    tailSum = p[iTail] + p[iTail + 1] + p[iTail + 2];
    break;
  default:
    SR_UNREACHABLE;
  }

  const __m256d *PTR_RESTRICT vp = SRCast::CPtr<__m256d>(p);
  __m256d sum;
  switch (nVects) {
  case 0:
    return tailSum;
  case 1:
    sum = _mm256_permute4x64_pd(Load<taCache>(vp), _MM_SHUFFLE(3, 1, 2, 0));
    break;
  default: {
    sum = Load<taCache>(vp);
    for (size_t i = 1, iEn=nVects-1; i < iEn; i++) {
      sum = HorizAddStraight(sum, Load<taCache>(vp + i));
    }
    sum = _mm256_hadd_pd(sum, Load<taCache>(vp + nVects - 1));
    break;
  } }

  const __m128d laneSums = _mm_hadd_pd(_mm256_extractf128_pd(sum, 1), _mm256_castpd256_pd128(sum));

  return laneSums.m128d_f64[0] + laneSums.m128d_f64[1] + tailSum;
}
FLOAT_PRECISE_END

template<typename T> struct SRSimd::CastImpl<T,T> {
  static T DoIt(const T par) { return par; }
};
template<> struct SRSimd::CastImpl<__m256d, __m256i> {
  static __m256d DoIt(const __m256i par) { return _mm256_castsi256_pd(par); }
};
template<> struct SRSimd::CastImpl<__m256i, __m256d> {
  static __m256i DoIt(const __m256d par) { return _mm256_castpd_si256(par); }
};

// For static class members, __vectorcall must be specified again in the definition: https://docs.microsoft.com/en-us/cpp/cpp/vectorcall
template<typename taResult, typename taParam> inline taResult __vectorcall SRSimd::Cast(const taParam par) {
  return CastImpl<taResult, taParam>::DoIt(par);
}

} // namespace SRPlat
