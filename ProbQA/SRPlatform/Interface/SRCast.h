// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

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
};

} // namespace SRPlat
