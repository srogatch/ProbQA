// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRCast.h"

namespace SRPlat {

template<typename taNumber> struct SRNumTraits;

template<> struct SRNumTraits<double> {
  static constexpr uint16_t _cnMantissaBits = 52;
  static constexpr uint16_t _cMantissaOffs = 0;
  static constexpr uint16_t _cnExponentBits = 11;
  static constexpr uint16_t _cExponentOffs = _cnMantissaBits;
  static constexpr uint16_t _cSignOffs = _cExponentOffs + _cnExponentBits;
  static constexpr uint16_t _cnTotalBits = _cnMantissaBits + _cnExponentBits + /* sign */ 1;

  static constexpr int16_t _cExponent0Down = 1023;
  static constexpr uint64_t _cExponent0Up = uint64_t(_cExponent0Down) << _cExponentOffs;
  static constexpr uint16_t _cExponentMaskDown = 0x7ff;
  static constexpr uint64_t _cExponentMaskUp = uint64_t(_cExponentMaskDown) << _cExponentOffs;
  static constexpr uint64_t _cSignMaskUp = 1ui64 << _cSignOffs;

  template<bool taNorm0> static int16_t ExtractExponent(const double value) {
    const int16_t exponent = _cExponentMaskDown & (SRCast::U64FromF64(value) >> _cExponentOffs);
    if (!taNorm0) {
      return exponent;
    }
    return exponent - _cExponent0Down;
  }
};

} // namespace SRPlat
