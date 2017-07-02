#pragma once

#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

// The array and its reserved size must be SIMD-aligned.
class SRBitHelper {
public: // constants
  static const size_t cLogSimdBits = 8;
  static const size_t cSimdBits = 1 << cLogSimdBits;
  static const size_t cSimdMask = cSimdBits - 1;

public:
  static size_t GetAlignedSizeBytes(const size_t nBits) {
    return ((nBits + cSimdMask) & (~cSimdMask)) >> 3;
  }
  template<bool taSkipCache> static void FillZero(__m256i *pArray, const size_t nBits) {
    const size_t nVects = (nBits + cSimdMask) >> cLogSimdBits;
    SRUtils::FillZeroVects<taSkipCache>(pArray, nVects);
  }
};

} // namespace SRPlat
