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
