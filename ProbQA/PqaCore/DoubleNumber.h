#pragma once

#include "../PqaCore/PqaNumber.h"

class DoubleNumber : public PqaNumber {
  double _value;
public:
  explicit DoubleNumber(double init) : _value(init) { }
  DoubleNumber& Mul(const DoubleNumber& fellow) { _value *= fellow._value; return *this; }
  DoubleNumber& Add(const DoubleNumber& fellow) { _value += fellow._value; return *this; }
};

static_assert(sizeof(DoubleNumber) == sizeof(double), "To allow AVX2 and avoid unaligned access penalties.");
