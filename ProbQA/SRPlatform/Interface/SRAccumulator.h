// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMacros.h"
#include "../SRPlatform/Interface/SRDoubleNumber.h"

namespace SRPlat {

// Kahan summator
template <typename taNumber> class SRAccumulator;

template<> class SRAccumulator<SRDoubleNumber> {
  double _sum;
  double _corr;

public:
  //SRAccumulator() { }
  explicit SRAccumulator(const SRDoubleNumber value) : _sum(value.GetValue()), _corr(0) { }
  inline SRDoubleNumber Get();
  inline SRAccumulator& Add(const SRDoubleNumber value);
};

FLOAT_PRECISE_BEGIN
inline SRAccumulator<SRDoubleNumber>& SRAccumulator<SRDoubleNumber>::Add(const SRDoubleNumber value) {
  const double y = value.GetValue() - _corr;
  const double t = _sum + y;
  _corr = (t - _sum) - y;
  _sum = t;
  return *this;
}
FLOAT_PRECISE_END

inline SRDoubleNumber SRAccumulator<SRDoubleNumber>::Get() {
  return SRDoubleNumber(_sum - _corr);
}

} // namespace SRPlat
