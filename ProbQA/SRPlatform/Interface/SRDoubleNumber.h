// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRRealNumber.h"
#include "../SRPlatform/Interface/SRCast.h"

namespace SRPlat {

class SRDoubleNumber : public SRRealNumber {
  double _value;
public:
  explicit SRDoubleNumber() { }
  explicit SRDoubleNumber(SRAmount init) : _value(SRCast::ToDouble(init)) { }

  double GetValue() const { return _value; }
  void SetValue(const double value) { _value = value; }

  SRDoubleNumber& Mul(const SRDoubleNumber& fellow) { 
    _value *= fellow._value;
    return *this;
  }
  SRDoubleNumber& Add(const SRDoubleNumber& fellow) {
    _value += fellow._value;
    return *this;
  }
  SRDoubleNumber operator*(const int64_t fellow) {
    SRDoubleNumber answer;
    answer._value = _value * fellow;
    return answer; 
  }
  SRDoubleNumber& operator+=(const SRAmount amount) {
    _value += SRCast::ToDouble(amount);
    return *this;
  }
};

static_assert(sizeof(SRDoubleNumber) == sizeof(double), "To allow AVX2 and avoid unaligned access penalties.");

} // namespace SRPlat
