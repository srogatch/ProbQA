// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRRealNumber.h"
#include "../SRPlatform/Interface/SRCast.h"
#include "../SRPlatform/Interface/SRPacked64.h"
#include "../SRPlatform/Interface/SRFastRandom.h"

namespace SRPlat {

class SRPLATFORM_API SRDoubleNumber : public SRRealNumber {
public: // constants
  static const int64_t _cMaxExp = 1023;
  static const int64_t _cExpOffs = 1023;

private: // variables
  double _value;

public:
  static __m128i __vectorcall ScaleBySizeBytesU32(const __m128i a);

  explicit SRDoubleNumber() { }
  explicit SRDoubleNumber(const SRAmount init) : _value(SRCast::ToDouble(init)) { }

  // Set to random number between 0 and |upper| inclusively. 
  static SRDoubleNumber MakeRandom(const SRDoubleNumber upper, SRFastRandom& fr) {
    SRDoubleNumber ans;
    ans._value = upper.GetValue() * fr.Generate<uint64_t>() / std::numeric_limits<uint64_t>::max();
    return ans;
  }

  SRAmount ToAmount() const { return _value; }

  double GetValue() const { return _value; }
  double& ModValue() { return _value; }
  void SetValue(const double value) { _value = value; }

  bool IsFinite() const { return std::isfinite(_value); }
  bool IsZero() const { return fabs(_value) == +0.0; }

  SRDoubleNumber& Mul(const SRDoubleNumber& fellow) { 
    _value *= fellow._value;
    return *this;
  }
  SRDoubleNumber& Add(const SRDoubleNumber& fellow) {
    _value += fellow._value;
    return *this;
  }
  SRDoubleNumber operator*(const int64_t fellow) const {
    SRDoubleNumber answer;
    answer._value = _value * fellow;
    return answer; 
  }
  SRDoubleNumber operator-(const SRDoubleNumber& fellow) const {
    SRDoubleNumber answer;
    answer._value = _value - fellow._value;
    return answer;
  }
  SRDoubleNumber& operator+=(const SRAmount amount) {
    _value += SRCast::ToDouble(amount);
    return *this;
  }
  SRDoubleNumber& operator+=(const SRDoubleNumber fellow) {
    _value += fellow._value;
    return *this;
  }

  bool operator<(const SRDoubleNumber fellow) const {
    return _value < fellow._value;
  }
  bool operator<=(const SRAmount fellow) const {
    return _value <= fellow;
  }
};

static_assert(sizeof(SRDoubleNumber) == sizeof(double), "To allow AVX2 and avoid unaligned access penalties.");

template<> struct SRNumPack<SRDoubleNumber> {
  static constexpr SRVectCompCount _cnComps = 4;
  __m256d _comps;

  SRNumPack() { }
  SRNumPack(const __m256d value) : _comps(value) { }
  void Set1(SRDoubleNumber value) { _comps = _mm256_set1_pd(value.GetValue()); }
};

static_assert(sizeof(SRNumPack<SRDoubleNumber>) == sizeof(__m256d), "To enable reinterpret_cast");

} // namespace SRPlat
