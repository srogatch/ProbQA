// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRMacros.h"

namespace SRPlat {

extern "C" {

// Although this just uses FPU instruction, it's slower than software approximation like std::log2().
SRPLATFORM_API ATTR_NOALIAS double __fastcall SRLog2MulD(const double toLog, const double toMul);

}

class SRMath {
public:
  // Works for non-negative only, and doesn't handle |factor==0| .
  template<typename T> static T RoundUpToFactor(const T num, const T factor) {
    const T a = num + factor - 1;
    return a - (a % factor);
  }
  // Works for non-negative only, and doesn't handle |factor==0| .
  template<typename T> static T RoundDownToFactor(const T num, const T factor) {
    return num - (num%factor);
  }

  // Seems to work for negatives to: it rounds negatives towards zero, i.e. up
  template<typename T> static T RShiftRoundUp(const T num, uint8_t nBits) {
    return (num + (T(1) << nBits) - 1) >> nBits;
  }

  ATTR_NOALIAS static uint8_t CeilLog2(const uint64_t val) {
    unsigned long index;
    const uint8_t overallMask = _BitScanReverse64(&index, val) ? 0xffui8 : 0ui8;
    //return index + ((val == (1ui64<<index)) ? 0 : 1);
    return overallMask & (uint8_t(index) + ((val & (val - 1)) ? 1ui8 : 0ui8));
  }

  // Computes pow(sqrt(2), p) very approximately: for odd p, the last multiplier is rather 1.5 than sqrt(2)
  ATTR_NOALIAS static uint64_t QuasiPowSqrt2(const uint8_t p) {
    uint64_t ans = (1ULL << (p >> 1));
    ans += ((-int64_t(p & 1)) & (ans >> 1));
    return ans;
  }

  ATTR_NOALIAS static uint8_t QuasiCeilLogSqrt2(const uint64_t val) {
    unsigned long index;
    const uint8_t overallMask = (val <= 1) ? 0ui8 : 0xffui8;
    _BitScanReverse64(&index, val); // no need to check the return value: it must be true for val>1
    const uint8_t baseLog = (uint8_t(index) << 1);
    const uint8_t halfCorr = (uint8_t(val >> (index - 1)) & 1ui8);
    const uint8_t fracCorr = (val & ~(3ui64 << (index - 1))) ? 1ui8 : 0ui8;
    return overallMask & (baseLog + halfCorr + fracCorr);
  }

  constexpr static uint8_t StaticFloorLog2(const size_t n) {
    return n  <= 1 ? 0ui8 : (StaticFloorLog2(n >> 1) + 1ui8);
  }
  constexpr static uint8_t StaticCeilLog2(const size_t n) {
    return n <= 1 ? 0ui8 : (StaticFloorLog2(n-1) + 1ui8);
  }
  constexpr static bool StaticIsPowOf2(const size_t n) {
    return (n & (n - 1)) == 0;
  }

};

} // namespace SRPlat
