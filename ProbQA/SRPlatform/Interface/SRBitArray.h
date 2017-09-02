// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRUtils.h"
#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRSimd.h"

namespace SRPlat {

// The array and its reserved size must be SIMD-aligned.
class SRBitHelper {
public: // constants
  static const size_t cSimdMask = SRSimd::_cNBits - 1;

public:
  ATTR_NOALIAS static size_t GetAlignedSizeBytes(const size_t nBits) {
    return ((nBits + cSimdMask) & (~cSimdMask)) >> 3;
  }

  template<bool taCache> ATTR_NOALIAS static void FillZero(__m256i *pArray, const size_t nBits) {
    const size_t nVects = SRSimd::VectsFromBits(nBits);
    SRUtils::FillZeroVects<taCache>(pArray, nVects);
  }
};

class SRBitArray {
  __m256i *_pBits;
  uint64_t _nBits : 56;
  uint64_t _comprCap : 8; // compressed capacity

private:
  ATTR_RESTRICT __m256i* ThrowingAlloc(uint64_t nPaddedBits) {
    assert((nPaddedBits & ((1ui64 << SRSimd::_cLogNBits) - 1)) == 0);
    return reinterpret_cast<__m256i*>(SRUtils::ThrowingSimdAlloc(nPaddedBits >> 3));
  }
public:
  explicit SRBitArray(const uint64_t nBits, const bool value = false) {
    // Allocate at least two __m256i values, so that next reallocation results in integer (3) number of values.
    _comprCap = SRMath::CompressCapacity<1ui64 << (SRSimd::_cLogNBits + 1)>(nBits);
    const uint64_t decomprCap = SRMath::DecompressCapacity(_comprCap);
    _pBits = ThrowingAlloc(decomprCap);
    _nBits = nBits;
    //TODO: initialize bits
  }
};

} // namespace SRPlat
