// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMath.h"

namespace SRPlat {

class SRSimd {
public:
  static constexpr uint8_t _cLogNBits = 8; //AVX2, 256 bits, log2(256)=8
  static constexpr uint8_t _cLogNBytes = _cLogNBits - 3;
  static constexpr size_t _cNBits = 1 << _cLogNBits;
  static constexpr size_t _cNBytes = 1 << _cLogNBytes;

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
};

} // namespace SRPlat
