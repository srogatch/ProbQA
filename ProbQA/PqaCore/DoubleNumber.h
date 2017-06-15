#pragma once

#include "../PqaCore/PqaNumber.h"

namespace ProbQA {

class DoubleNumber : public PqaNumber {
  double _value;
public:
  explicit DoubleNumber() { }
  explicit DoubleNumber(TPqaAmount init) : _value(to_double(init)) { }

  double GetValue() { return _value; }
  void SetValue(const double value) { _value = value; }

  DoubleNumber& Mul(const DoubleNumber& fellow) { 
    _value *= fellow._value;
    return *this;
  }
  DoubleNumber& Add(const DoubleNumber& fellow) {
    _value += fellow._value;
    return *this;
  }
  DoubleNumber operator*(const int64_t fellow) {
    DoubleNumber answer;
    answer._value = _value * fellow;
    return answer; 
  }

};

static_assert(sizeof(DoubleNumber) == sizeof(double), "To allow AVX2 and avoid unaligned access penalties.");

} // namespace ProbQA
