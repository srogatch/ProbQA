// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRBasicTypes.h"

namespace SRPlat {

class SRCast {
public:
  // Helper method to catch overflow error
  template<typename T> static size_t ToSizeT(const T val) {
    return static_cast<size_t>(val);
  }
  template<typename T> static uint64_t ToUint64(const T val) {
    return static_cast<uint64_t>(val);
  }
  static double ToDouble(const SRAmount amount) { return amount; }

  static uint64_t CastF64_U64(const double value) {
    // _castf64_u64() doesn't seem present in MSVC++
    return *reinterpret_cast<const uint64_t*>(&value);
  }
};

} // namespace SRPlat
