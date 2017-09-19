// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMacros.h"

namespace SRPlat {

class SRAccumulator {
  double _sum;
  double _corr;

public:
  SRAccumulator() { }
  explicit SRAccumulator(const double value) : _sum(value), _corr(0) { }
  inline double Get();
  inline SRAccumulator& Add(const double value);
};

FLOAT_PRECISE_BEGIN
inline SRAccumulator& SRAccumulator::Add(const double value) {
  const double y = value - _corr;
  const double t = _sum + y;
  _corr = (t - _sum) - y;
  _sum = t;
  return *this;
}
FLOAT_PRECISE_END

inline double SRAccumulator::Get() {
  return _sum - _corr;
}

} // namespace SRPlat
