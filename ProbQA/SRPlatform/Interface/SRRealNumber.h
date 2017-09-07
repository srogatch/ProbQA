// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRAmount.h"

namespace SRPlat {

class SRRealNumber {
protected:
  SRRealNumber() { }

public:
  explicit SRRealNumber(SRAmount init);
  SRRealNumber& Mul(const SRRealNumber& fellow);
  SRRealNumber& Add(const SRRealNumber& fellow);
  SRRealNumber& operator+=(const SRRealNumber amount);
};

template <typename taNumber> struct SRNumPack;

} // namespace SRPlat
