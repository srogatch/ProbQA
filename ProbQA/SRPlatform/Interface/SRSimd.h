// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRNumTraits.h"

namespace SRPlat {

class SRPLATFORM_API SRSimd {
  template<typename taResult, typename taParam> struct CastImpl;

public:
  typedef __m256i TIntSimd;
  union Packed64 {
    double _f64;
    float _f32[2];
    uint64_t _u64;
    int64_t _i64;
    uint32_t _u32[2];
    int32_t _i32[2];
    uint16_t _u16[4];
    int16_t _i16[4];
    uint8_t _u8[8];
    int8_t _i8[8];

    Packed64() { }

    Packed64(const __m128i& vect, const uint8_t at) : _u64(vect.m128i_u64[at]) { }
    Packed64(const __m128& vect, const uint8_t at) : Packed64(_mm_castps_si128(vect), at) { }
    Packed64(const __m128d& vect, const uint8_t at) : Packed64(_mm_castpd_si128(vect), at) { }
    //template<uint8_t taAt> explicit Packed64(const __m128i& vect, const uint8_t taAt)
    //  : _i64(_mm_extract_epi64(vect, taAt)) { }
    constexpr explicit Packed64(const uint64_t value) : _u64(value) { }
    constexpr static Packed64 Set1U32(const uint32_t value) {
      return Packed64((uint64_t(value) << 32) | value);
    }
  };

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
    const __m256i *const genP = reinterpret_cast<const __m256i*>(p);
    //TODO: verify that taCache based branchings are compile-time
    const __m256i ans = (taCache ? _mm256_load_si256(genP) : _mm256_stream_load_si256(genP));
    return Cast<taVect>(ans);
  }

  template<bool taCache, typename taVect> static std::enable_if_t<sizeof(taVect)==sizeof(__m256i)> __vectorcall
  Store(taVect *p, const taVect v)
  {
    __m256i *genP = reinterpret_cast<__m256i*>(p);
    //TODO: verify that this casting turns into no-op in assembly, so that the value just stays in the register
    const __m256i genV = Cast<__m256i>(v);
    taCache ? _mm256_store_si256(genP, genV) : _mm256_stream_si256(genP, genV);
  }

  static size_t GetPaddedBytes(const size_t nUnpaddedBytes) {
    return (nUnpaddedBytes + _cByteMask) & (~_cByteMask);
  }

  template<size_t taItemSize> static size_t PaddedBytesFromItems(const size_t nItems) {
    return GetPaddedBytes(nItems * taItemSize);
  }

  ATTR_NOALIAS static __m256i __vectorcall SetMsb1(const uint16_t nMsb1) {
    const __m256i ones = _mm256_set1_epi8(-1i8);
    __m256i shift = _mm256_set1_epi32(nMsb1);
    shift = _mm256_subs_epu16(_cSet1MsbOffs, shift);
    return _mm256_sllv_epi32(ones, shift);
  }

  ATTR_NOALIAS static __m256i __vectorcall SetLsb1(const uint16_t nLsb1) {
    const __m256i ones = _mm256_set1_epi8(-1i8);
    __m256i shift = _mm256_set1_epi32(nLsb1);
    shift = _mm256_subs_epu16(_cSet1LsbOffs, shift);
    return _mm256_srlv_epi32(ones, shift);
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

  template<bool taNorm0> ATTR_NOALIAS static Packed64 __vectorcall ExtractExponents32(const __m128d nums) {
    const __m128i shuffled = _mm_shuffle_epi32(_mm_castpd_si128(nums), _MM_SHUFFLE(0, 0, 3, 1));
    const Packed64 high32(shuffled, 0);
    constexpr Packed64 expMask = Packed64::Set1U32(SRNumTraits<double>::_cExponentMaskDown);
    const Packed64 exps((high32._u64 >> (SRNumTraits<double>::_cExponentOffs - 32)) & expMask._u64);
    if (!taNorm0) {
      return exps;
    }
    constexpr Packed64 normalizer = Packed64::Set1U32(SRNumTraits<double>::_cExponent0Down);
    return Packed64(exps._u64 - normalizer._u64);
  }

  ATTR_NOALIAS static __m256d __vectorcall MakeExponent0(const __m256d nums) {
    const __m256d e0nums = _mm256_or_pd(_mm256_castsi256_pd(_cDoubleExp0Up),
      _mm256_andnot_pd(_mm256_castsi256_pd(_cDoubleExpMaskUp), nums));
    return e0nums;
  }
};

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
