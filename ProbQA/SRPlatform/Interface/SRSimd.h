// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMath.h"

namespace SRPlat {

class SRSimd {
  template<typename taResult, typename taParam> struct CastImpl;

public:
  typedef __m256i TIntSimd;

  static constexpr uint8_t _cLogNBits = 8; //AVX2, 256 bits, log2(256)=8
  static constexpr uint8_t _cLogNBytes = _cLogNBits - 3;
  static constexpr size_t _cNBits = 1 << _cLogNBits;
  static constexpr size_t _cNBytes = 1 << _cLogNBytes;

  static constexpr size_t _cByteMask = _cNBytes - 1;
  static constexpr size_t _cBitMask = _cNBits - 1;

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
