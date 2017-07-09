// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace SRPlat {

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

  constexpr static size_t StaticFloorLog2(const size_t n) {
    return n  <= 1 ? 0 : (StaticFloorLog2(n >> 1) + 1);
  }
  constexpr static size_t StaticCeilLog2(const size_t n) {
    return n <= 1 ? 0 : (StaticFloorLog2(n-1)+1);
  }

};

} // namespace SRPlat
