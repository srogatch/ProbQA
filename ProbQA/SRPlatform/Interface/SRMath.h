// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace SRPlat {

class SRMath {
public:
  // Works for positive only, and doesn't handle |factor==0| .
  template<typename T> static T RoundUpToFactor(const T num, const T factor) {
    const T a = num + factor - 1;
    return a - (a % factor);
  }
};

} // namespace SRPlat
