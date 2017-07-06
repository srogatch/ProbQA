// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

class PqaNumber {
protected:
  PqaNumber() { }

public:
  PqaNumber(TPqaAmount init);
  PqaNumber& Mul(const PqaNumber& fellow);
  PqaNumber& Add(const PqaNumber& fellow);
  PqaNumber& operator+=(const TPqaAmount amount);
};

} // namespace ProbQA
