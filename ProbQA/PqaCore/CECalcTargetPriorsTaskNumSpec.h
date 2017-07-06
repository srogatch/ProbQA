// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/DoubleNumber.h"

namespace ProbQA {

// Number-specific data for CECalcTargetPriorsTask
template <typename taNumber> class CECalcTargetPriorsTaskNumSpec;

template<> class CECalcTargetPriorsTaskNumSpec<DoubleNumber> {
public: // variables
  __m256d _divisor;
};

} // namespace ProbQA
