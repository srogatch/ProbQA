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

  template<bool taCache> ATTR_NOALIAS static void FillZero(__m256i *PTR_RESTRICT pArray, const int64_t nBits) {
    const size_t nVects = SRSimd::VectsFromBits(nBits);
    SRUtils::FillZeroVects<taCache>(pArray, nVects);
  }

  ATTR_NOALIAS static bool Test(const __m256i *PTR_RESTRICT pArray, const int64_t iBit) {
    //return SRCast::CPtr<uint8_t>(pArray)[iBit >> 3] & (1ui8 << (iBit & 7));
    return _bittest64(SRCast::CPtr<int64_t>(pArray), iBit) != 0;
  }

  // Returns the previous state of the bit.
  static bool Set(__m256i *pArray, const int64_t iBit) {
    return _bittestandset64(SRCast::Ptr<int64_t>(pArray), iBit) != 0;
  }

  template<typename taResult> static const taResult& GetPacked(const __m256i *pArray, const uint64_t iPack) {
    return SRCast::CPtr<taResult>(pArray)[iPack];
  }
};

class SRBitArray {
  static constexpr size_t _cMinVects = 2;
  __m256i *_pBits;
  uint64_t _nBits : 56;
  uint64_t _comprCap : 7; // compressed capacity, in vectors
  const uint64_t _defaultVal : 1;

private:
  ATTR_RESTRICT static __m256i* ThrowingAlloc(const size_t nVects) {
    return SRCast::Ptr<__m256i>(SRUtils::ThrowingSimdAlloc(nVects << SRSimd::_cLogNBytes));
  }

  template<typename taMaskVisit, typename taFullVisit> void VisitRange(const uint64_t iFirst, const uint64_t iLim,
    const taMaskVisit& maskVisit, const taFullVisit& fullVisit)
  {
    assert(iFirst <= iLim);
    size_t i256First = iFirst >> SRSimd::_cLogNBits;
    size_t i256Lim = iLim >> SRSimd::_cLogNBits;
    const __m256i firstMask = SRSimd::SetHighBits1(SRSimd::_cNBits - (iFirst & SRSimd::_cBitMask));
    if(i256Lim <= i256First) {
      const __m256i limMask = SRSimd::SetLowBits1(iLim & SRSimd::_cBitMask);
      maskVisit(_pBits[i256First], _mm256_and_si256(firstMask, limMask));
      return;
    }
    maskVisit(_pBits[i256First], firstMask);
    for(size_t i= i256First+1; i<i256Lim; i++) {
      fullVisit(_pBits[i]);
    }
    // Prevent visiting out of bounds if iLim is right beyond the end of the array.
    if(iLim & SRSimd::_cBitMask) {
      const __m256i limMask = SRSimd::SetLowBits1(iLim & SRSimd::_cBitMask);
      maskVisit(_pBits[i256Lim], limMask);
    }
  }

  template<bool taOppVal> void GrowInternal(const uint64_t newNBits) {
    const uint64_t capBits = SRMath::DecompressCapacity<SRSimd::_cNBits>(_comprCap);
    assert((capBits & SRSimd::_cBitMask) == 0);
    const uint64_t oldNBits = _nBits;
    if (newNBits <= capBits) {
      if constexpr (taOppVal) {
        AssignRange(!_defaultVal, oldNBits, newNBits);
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

    if constexpr (taOppVal) {
      const uint64_t newCapBits = uint64_t(newCapVects) << SRSimd::_cLogNBits;
      //AssignRange(!_defaultVal, oldNBits, newNBits);
      //AssignRange(_defaultVal, newNBits, newCapVects << SRSimd::_cLogNBits);
      _defaultVal ?
        (ClearRange(oldNBits, newNBits), SetRange(newNBits, newCapBits))
        : (SetRange(oldNBits, newNBits), ClearRange(newNBits, newCapBits));
    } else {
      // Default-initialize the bits
      const __m256i initVect = _mm256_set1_epi8(-int8_t(_defaultVal));
      for (size_t i = oldCapVects; i<newCapVects; i++) {
        _mm256_store_si256(_pBits + i, initVect);
      }
    }
  }

public:
  explicit SRBitArray(const uint64_t nBits, const bool defaultVal = false) : _defaultVal(defaultVal ? 1 : 0) {
    // Allocate at least two __m256i values, so that next reallocation results in integer (3) number of values.
    _comprCap = SRMath::CompressCapacity<_cMinVects>(SRMath::RShiftRoundUp(nBits, SRSimd::_cLogNBits));
    const size_t capVects = SRMath::DecompressCapacity<1>(_comprCap);
    _pBits = ThrowingAlloc(capVects);
    _nBits = nBits;
    const __m256i initVect = _mm256_set1_epi8(-int8_t(_defaultVal));
    for(size_t i=0; i<capVects; i++) {
      _mm256_store_si256(_pBits + i, initVect);
    }
  }

  //// Append tail bits
  void Add(const uint64_t nToAdd) {
    GrowTo(_nBits + nToAdd);
  }
  void Add(const uint64_t nToAdd, const bool value) {
    GrowTo(_nBits + nToAdd, value);
  }

  void GrowTo(const uint64_t targNBits) {
    GrowInternal<false>(targNBits);
  }
  void GrowTo(const uint64_t targNBits, const bool value) {
    (value == bool(_defaultVal)) ? GrowInternal<false>(targNBits) : GrowInternal<true>(targNBits);
  }

  // Remove tail bits
  void Remove(const uint64_t nToRemove) {
    assert(nToRemove <= _nBits);
    ReduceTo(_nBits - nToRemove);
  }

  void ReduceTo(const uint64_t targNBits) {
    assert(targNBits <= _nBits);
    AssignRange(_defaultVal, targNBits, _nBits);
    _nBits = targNBits;
  }

  void ShrinkTo(const uint64_t nBits) {
    assert(nBits <= _nBits);
    const uint8_t newComprCap = SRMath::CompressCapacity<_cMinVects>(
      SRMath::RShiftRoundUp(nBits, SRSimd::_cLogNBits));
    if(newComprCap >= _comprCap) {
      assert(newComprCap == _comprCap);
      AssignRange(_defaultVal, nBits, _nBits);
      _nBits = nBits;
      return;
    }
    const size_t newCapVects = SRMath::DecompressCapacity<1>(newComprCap);
    __m256i *pNewBits = ThrowingAlloc(newCapVects);

    const size_t nUsedVects = SRMath::RShiftRoundUp(nBits, SRSimd::_cLogNBits);
    SRUtils::Copy256<true, false>(pNewBits, _pBits, nUsedVects);
    AssignRange(_defaultVal, nBits, newCapVects << SRSimd::_cLogNBits);

    const size_t oldCapVects = SRMath::DecompressCapacity<1>(_comprCap);
    SRUtils::FlushCache<false, false>(_pBits, oldCapVects << SRSimd::_cLogNBytes);
    _mm_free(_pBits);

    _pBits = pNewBits;
    _comprCap = newComprCap;
    _nBits = nBits;
  }

  //// A group of methods for fast single-bit manipulations
  // Returns the previous state of this bit.
  bool SetOne(const uint64_t iBit) {
    //SRCast::Ptr<uint8_t>(_pBits)[iBit >> 3] |= (1ui8 << (iBit & 7));
    return _bittestandset64(SRCast::Ptr<int64_t>(_pBits), iBit) != 0;
  }
  // Returns the previous state of this bit.
  bool ClearOne(const uint64_t iBit) {
    //SRCast::Ptr<uint8_t>(_pBits)[iBit >> 3] &= ~(1ui8 << (iBit & 7));
    return _bittestandreset64(SRCast::Ptr<int64_t>(_pBits), iBit) != 0;
  }
  // Returns the previous state of this bit.
  bool ToggleOne(const uint64_t iBit) {
    //SRCast::Ptr<uint8_t>(_pBits)[iBit >> 3] ^= (1ui8 << (iBit & 7));
    return _bittestandcomplement64(SRCast::Ptr<int64_t>(_pBits), iBit) != 0;
  }

  //// Multi-bit manipulations
  void SetRange(const uint64_t iFirst, const uint64_t iLim) {
    VisitRange(iFirst, iLim,
      [](__m256i& val, const __m256i& mask) {
        val = _mm256_or_si256(val, mask);
      },
      [](__m256i& val) {
        val = _mm256_set1_epi8(-1i8);
      }
    );
  }

  void ClearRange(const uint64_t iFirst, const uint64_t iLim) {
    VisitRange(iFirst, iLim,
      [](__m256i& val, const __m256i& mask) {
        val = _mm256_andnot_si256(mask, val);
      },
      [](__m256i& val) {
        val = _mm256_setzero_si256();
      }
    );
  }

  void ToggleRange(const uint64_t iFirst, const uint64_t iLim) {
    VisitRange(iFirst, iLim,
      [](__m256i& val, const __m256i& mask) {
        val = _mm256_xor_si256(mask, val);
      },
      [](__m256i& val) {
        val = _mm256_xor_si256(val, _mm256_set1_epi8(-1i8));
      }
    );
  }

  void AssignRange(const bool value, const uint64_t iFirst, const uint64_t iLim) {
    value ? SetRange(iFirst, iLim) : ClearRange(iFirst, iLim);
  }

  bool GetOne(const uint64_t iBit) const {
    //return SRCast::CPtr<uint8_t>(_pBits)[iBit >> 3] & (1ui8 << (iBit & 7));
    return _bittest64(SRCast::CPtr<int64_t>(_pBits), iBit) != 0;
  }

  uint8_t GetQuad(const uint64_t iQuad) const {
    const uint8_t packed = SRCast::CPtr<uint8_t>(_pBits)[iQuad >> 1];
    const uint8_t shift = (iQuad & 1) << 2;
    return (packed>>shift) & 0x0f;
  }

  template<typename taResult> const taResult& GetPacked(const uint64_t iPack) const {
    return SRCast::CPtr<taResult>(_pBits)[iPack];
  }

  uint64_t Size() const {
    return _nBits;
  }

  void* Data() const {
    return _pBits;
  }
};

} // namespace SRPlat
