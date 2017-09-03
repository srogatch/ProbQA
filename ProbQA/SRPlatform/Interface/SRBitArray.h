// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRUtils.h"
#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRCast.h"

namespace SRPlat {

// The array and its reserved size must be SIMD-aligned.
class SRBitHelper {
public:
  ATTR_NOALIAS static size_t GetAlignedSizeBytes(const size_t nBits) {
    return ((nBits + SRSimd::_cBitMask) & (~SRSimd::_cBitMask)) >> 3;
  }

  template<bool taCache> ATTR_NOALIAS static void FillZero(__m256i *pArray, const size_t nBits) {
    const size_t nVects = SRSimd::VectsFromBits(nBits);
    SRUtils::FillZeroVects<taCache>(pArray, nVects);
  }
};

class SRBitArray {
  static constexpr size_t _cMinVects = 2;
  __m256i *_pBits;
  uint64_t _nBits : 56;
  uint64_t _comprCap : 7; // compressed capacity, in vectors
  uint64_t _defaultVal : 1;

private:
  ATTR_RESTRICT static __m256i* ThrowingAlloc(const size_t nVects) {
    return reinterpret_cast<__m256i*>(SRUtils::ThrowingSimdAlloc(nVects << SRSimd::_cLogNBytes));
  }

  template<bool taHaveCap> void ToggleInternal(const uint64_t iFirst, const uint64_t iLim, const uint64_t capBits = 0) {
    size_t i64First = iFirst >> 6;
    size_t i64Lim = iLim >> 6;
    const uint64_t firstMask = -(1ui64 << (iFirst & 63));
    const uint64_t limMask = (1ui64 << (iLim & 63)) - 1;
    // Also provide an early exit for the frequent case of adding/toggling one bit
    if (i64First == i64Lim) {
      reinterpret_cast<uint64_t*>(_pBits)[i64First] ^= (firstMask & limMask);
      return;
    }
    reinterpret_cast<uint64_t*>(_pBits)[i64First] ^= firstMask;
    i64First++;
    assert(!taHaveCap || ((capBits > 0) && ((capBits & 255) == 0)));
    // Prevent XORring out of bounds if iLim is right beyond the end of the array.
    if((taHaveCap && iLim < capBits) || (!taHaveCap && limMask != 0)) {
      reinterpret_cast<uint64_t*>(_pBits)[i64Lim] ^= limMask;
    }
    const size_t i256First = SRMath::RShiftRoundUp(i64First, 2);
    const size_t i256Lim = i64Lim >> 2;
    if(i256First >= i256Lim) {
      for(size_t i=i64First; i<i64Lim; i++) {
        reinterpret_cast<int64_t*>(_pBits)[i] ^= -1i64;
      }
      return;
    }
    for (size_t i = i64First, iEn = i256First << 2; i < iEn; i++) {
      reinterpret_cast<int64_t*>(_pBits)[i] ^= -1i64;
    }
    const __m256i vectOnes = _mm256_set1_epi8(-1i8);
    for(size_t i=i256First; i<i256Lim; i++) {
      _mm256_store_si256(_pBits + i, _mm256_xor_si256(_mm256_load_si256(_pBits + i), vectOnes));
    }
    for(size_t i=i256Lim<<2; i<i64Lim; i++) {
      reinterpret_cast<int64_t*>(_pBits)[i] ^= -1i64;
    }
  }

  template<bool taOppVal> void AddInternal(const uint64_t nBits) {
    const uint64_t capBits = SRMath::DecompressCapacity<SRSimd::_cNBits>(_comprCap);
    assert((capBits & SRSimd::_cBitMask) == 0);
    const uint64_t oldNBits = _nBits;
    const uint64_t newNBits = oldNBits + nBits;
    if (newNBits <= capBits) {
      if(taOppVal) {
        ToggleInternal<true>(oldNBits, newNBits, capBits);
      } // otherwise the bits must be already initialized to default value
      _nBits = newNBits;
      return;
    }
    // This should be faster than a loop incrementing _comprCap and checking if decompressed value is at least newBits
    const uint8_t newComprCap = SRMath::CompressCapacity<_cMinVects>(
      SRMath::RShiftRoundUp(newNBits, SRSimd::_cLogNBits));

    // Realloc the array
    const size_t newCapVects = SRMath::DecompressCapacity<1>(newComprCap);
    __m256i *pNewBits = ThrowingAlloc(newCapVects);
    const size_t oldCapVects = capBits >> SRSimd::_cLogNBits;
    SRUtils::Copy256<true, false>(pNewBits, _pBits, oldCapVects);
    SRUtils::FlushCache<false, false>(_pBits, oldCapVects << SRSimd::_cLogNBytes);
    _mm_free(_pBits);

    _pBits = pNewBits;
    _comprCap = newComprCap;
    _nBits = newNBits;

    // Default-initialize the bits
    const __m256i initVect = _mm256_set1_epi8(-int8_t(_defaultVal));
    for (size_t i = oldCapVects; i<newCapVects; i++) {
      _mm256_store_si256(_pBits + i, initVect);
    }

    if(taOppVal) {
      // Changing from "setDefault,toggle" to "setTarget" would complicate the implementation and could be slower for
      //   small nBits values.
      ToggleInternal<true>(oldNBits, newNBits, newCapVects << SRSimd::_cLogNBits);
    }
  }

public:
  explicit SRBitArray(const uint64_t nBits, const bool defaultVal = false) {
    // Allocate at least two __m256i values, so that next reallocation results in integer (3) number of values.
    _comprCap = SRMath::CompressCapacity<_cMinVects>(SRMath::RShiftRoundUp(nBits, SRSimd::_cLogNBits));
    const size_t capVects = SRMath::DecompressCapacity<1>(_comprCap);
    _pBits = ThrowingAlloc(capVects);
    _nBits = nBits;
    _defaultVal = defaultVal ? 1 : 0;
    const __m256i initVect = _mm256_set1_epi8(-int8_t(_defaultVal));
    for(size_t i=0; i<capVects; i++) {
      _mm256_store_si256(_pBits + i, initVect);
    }
  }

  void Add(const uint64_t nBits) {
    AddInternal<false>(nBits);
  }
  void Add(const uint64_t nBits, const bool value) {
    (value == _defaultVal) ? AddInternal<false>(nBits) : AddInternal<true>(nBits);
  }

};

} // namespace SRPlat
