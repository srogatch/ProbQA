// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

//TODO: rename because it now specifies for both a parallel train task and just a sequential train operation
// Number-specific data for CETrainTask
template <typename taNumber> class CETrainTaskNumSpec;

template<> class CETrainTaskNumSpec<SRPlat::SRDoubleNumber> {
public: // variables
  double _inc2B;
  double _incBSquare;
  double _inc4B;
  double _inc2BSquare;

public: // methods

  explicit CETrainTaskNumSpec(const TPqaAmount amount) {
    // (a+b)**2 = a**2 + 2*a*b + b**2
    const double b = SRPlat::SRCast::ToDouble(amount);
    //TODO: check that optimizer uses SSE/AVX here, otherwise rewrite manually
    _inc2B = 2 * b;
    _inc2BSquare = b * b;
    _inc4B = 4 * b;
    _inc2BSquare = 4 * _inc2BSquare;
  }
};

} // namespace ProbQA
