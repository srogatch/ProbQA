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

  // For positives only
  template<typename T> static T PosDivideRoundUp(const T num, const T divisor) {
    assert(num >= 0);
    return (num + divisor - 1) / divisor;
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

  // Computes pow(sqrt(2), p) very approximately: for odd p, the last multiplier is rather 1.5 than sqrt(2).
  // Returns 1 for p==0 and p==1.
  ATTR_NOALIAS static uint64_t QuasiPowSqrt2(const uint8_t p) {
    uint64_t ans = (1ULL << (p >> 1));
    ans += ((-int64_t(p & 1)) & (ans >> 1));
    return ans;
  }

  // Returns 0 for val==0. Returns 1 for val==1.
  ATTR_NOALIAS static uint8_t QuasiCeilLogSqrt2(const uint64_t val) {
    unsigned long index;
    const uint8_t overallMask = _BitScanReverse64(&index, val) ? 0xffui8 : 0ui8;
    const uint8_t baseLog = (uint8_t(index) << 1);
    const uint8_t im1 = uint8_t(index) - 1;
    const uint8_t halfCorr = (uint8_t(val >> im1) & 1ui8);
    const uint8_t fracCorr = (val & ((1ui64<<im1)-1)) ? 1ui8 : 0ui8;
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

  // Computes pow(sqrt(2), p) very approximately: for odd p, the last multiplier is rather 1.5 than sqrt(2).
  // For p<=1 when nPacked is not 1 or even number, it may return a number that is not a multiple of nPacked.
  template<uint64_t taNPacked> ATTR_NOALIAS static uint64_t DecompressCapacity(const uint8_t p) {
    uint64_t ans = (taNPacked << (p >> 1));
    ans += ((-int64_t(p & 1)) & (ans >> 1));
    return ans;
  }

  template<uint64_t taMinVal> ATTR_NOALIAS static uint8_t CompressCapacity(uint64_t val) {
    static_assert(taMinVal >= 2, "It's important to prevent val<=1 from leaking to subsequent code.");
    // Also it's important to avoid returning 0, because by incrementing this compressed capacity to 1, the client
    //   code will not get a greater uncompressed capacity. So the options here are: (val<=2) ret 2; (val <= 3) ret 3;
    //   (val<=4) ret 4; (val<=6) ret 5; (val<=8) ret 6;

    // Although this doesn't recompute the constant to return, this makes branching which seems worse.
    //if (val <= taMin) {
    //  // Return precomputed compressed capacity for uncompressed capacity taMin.
    //  constexpr unsigned long index = StaticFloorLog2(taMin);
    //  constexpr uint8_t baseLog = (uint8_t(index) << 1);
    //  constexpr uint8_t im1 = uint8_t(index) - 1;
    //  constexpr uint8_t halfCorr = (uint8_t(taMin >> im1) & 1ui8);
    //  constexpr uint8_t fracCorr = (taMin & ((1ui64 << im1) - 1)) ? 1ui8 : 0ui8;
    //  return baseLog + halfCorr + fracCorr;
    //}

    //val = std::max(val, taMinVal);
    if(val < taMinVal) {
      val = taMinVal;
    }
    unsigned long index;
    _BitScanReverse64(&index, val);
    const uint8_t baseLog = (uint8_t(index) << 1);
    const uint8_t im1 = uint8_t(index) - 1;
    const uint8_t halfCorr = (uint8_t(val >> im1) & 1ui8);
    const uint8_t fracCorr = (val & ((1ui64 << im1) - 1)) ? 1ui8 : 0ui8;
    return baseLog + halfCorr + fracCorr;
  }
};

} // namespace SRPlat
