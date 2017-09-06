// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

// Number-specific data for CETrainTask
template <typename taNumber> class CETrainTaskNumSpec;

template<> class CETrainTaskNumSpec<SRPlat::SRDoubleNumber> {
public: // variables
  __m256d _fullAddend; // non-colliding (4 at once)
  __m256d _collAddend; // colliding (3 at once)
};

} // namespace ProbQA
