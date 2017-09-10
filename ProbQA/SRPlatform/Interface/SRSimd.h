// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRNumTraits.h"
#include "../SRPlatform/Interface/SRPacked64.h"

namespace SRPlat {

class SRPLATFORM_API SRSimd {
  template<typename taResult, typename taParam> struct CastImpl;

public:
  typedef __m256i TIntSimd;

  static constexpr uint8_t _cLogNBits = 8; //AVX2, 256 bits, log2(256)=8
  static constexpr uint8_t _cLogNBytes = _cLogNBits - 3;
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
private:
  static const __m256i _cSet1MsbOffs;
  static const __m256i _cSet1LsbOffs;
  static constexpr uint8_t _cnStbqEntries = 1 << 4;
  static const uint32_t _cStbqTable[_cnStbqEntries];

public:
  static size_t VectsFromBytes(const size_t nBytes) {
    return SRMath::RShiftRoundUp(nBytes, _cLogNBytes);
  }
  static size_t VectsFromBits(const size_t nBits) {
    return SRMath::RShiftRoundUp(nBits, _cLogNBits);
  }

  template<typename taComp> static size_t VectsFromComps(const size_t nComps) {
    static_assert(sizeof(taComp) <= _cNBytes, "Component must no larger than SIMD vector.");
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

  template<bool taNorm0> ATTR_NOALIAS static __m128i __vectorcall ExtractExponents32(const __m256d nums) {
    const __m128 hiLane = _mm_castpd_ps(_mm256_extractf128_pd(nums, 1));
    const __m128 loLane = _mm_castpd_ps(_mm256_castpd256_pd128(nums));
    const __m128i high32 = _mm_castps_si128(_mm_shuffle_ps(loLane, hiLane, _MM_SHUFFLE(3, 1, 3, 1)));
    const __m128i exps = _mm_and_si128(_cDoubleExpMaskDown32,
      _mm_srli_epi32(high32, SRNumTraits<double>::_cExponentOffs - 32));
    if (!taNorm0) {
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

  ATTR_NOALIAS static __m256d __vectorcall HorizAddStraight(const __m256d a, const __m256d b) {
    const __m256d crossed = _mm256_hadd_pd(a, b);
    const __m256d straight = _mm256_permute4x64_pd(crossed, _MM_SHUFFLE(3, 1, 2, 0));
    return straight;
  }

  ATTR_NOALIAS static __m256i __vectorcall MaxI64(const __m256i a, const __m256i b, const __m256i maskA) {
    const __m256i isAGreater = _mm256_cmpgt_epi64(a, b);
    const __m256i maxes = _mm256_blendv_epi8(b, a, _mm256_or_si256(isAGreater, maskA));
    return maxes;
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

  template<bool taCache> ATTR_NOALIAS inline static double StableSum(double *p, const size_t nDoubles);
};

template<bool taCache> ATTR_NOALIAS inline double SRSimd::StableSum(double *p, const size_t nDoubles) {
  const size_t nVects = nDoubles >> 2;
  const size_t iTail = nVects << 2;
  const SRVectCompCount nTail = nDoubles - iTail;
  double tailSum;

  FLOAT_PRECISE_BEGIN;
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
  FLOAT_PRECISE_END;

  const __m256d *vp = SRCast::CPtr<__m256d>(p);
  __m256d sum;
  switch (nVects) {
  case 0:
    return tailSum;
  case 1:
    sum = _mm256_permute4x64_pd(Load<taCache>(vp), _MM_SHUFFLE(3, 1, 2, 0));
    break;
  default: {
    sum = Load<taCache>(vp);
    for (size_t i = 1; i + 1 < nVects; i++) {
      sum = HorizAddStraight(sum, vp[i]);
    }
    sum = _mm256_hadd_pd(sum, vp[nVects - 1]);
    break;
  } }

  const __m128d laneSums = _mm_hadd_pd(_mm256_extractf128_pd(sum, 1), _mm256_castpd256_pd128(sum));

  FLOAT_PRECISE_BEGIN;
  return laneSums.m128d_f64[0] + laneSums.m128d_f64[1] + tailSum;
  FLOAT_PRECISE_END;
}

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
