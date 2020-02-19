// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRBasicTypes.h"
#include "../SRPlatform/Interface/SRMacros.h"

namespace SRPlat {

class SRCast {
public:
  // Helper method to catch overflow error
  template<typename T> ATTR_NOALIAS constexpr static size_t ToSizeT(const T val) {
    return static_cast<size_t>(val);
  }

  template<typename T> ATTR_NOALIAS constexpr static uint64_t ToUint64(const T val) {
    return static_cast<uint64_t>(val);
  }

  template<typename T> ATTR_NOALIAS constexpr static int32_t ToInt32(const T val) {
    return static_cast<int32_t>(val);
  }

  template<typename taResult, typename taParam> ATTR_NOALIAS constexpr static taResult* Ptr(taParam *p) {
    return reinterpret_cast<taResult*>(p);
  }

  template<typename taResult, typename taParam> ATTR_NOALIAS constexpr static const taResult* CPtr(const taParam *p) {
    return reinterpret_cast<const taResult*>(p);
  }

  template<typename taResult, typename taParam> ATTR_NOALIAS constexpr static const taResult& Bitwise(
    const taParam& value)
  {
    return *CPtr<taResult>(&value);
  }

  ATTR_NOALIAS static double ToDouble(const SRAmount amount) { return amount; }

  ATTR_NOALIAS static uint64_t U64FromF64(const double value) {
    // _castf64_u64() doesn't seem present in MSVC++
    return *reinterpret_cast<const uint64_t*>(&value);
  }
  ATTR_NOALIAS static double F64FromU64(const uint64_t value) {
    return *reinterpret_cast<const double*>(&value);
  }
};

} // namespace SRPlat
