// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CERadixSortRatingsTask.h"

namespace ProbQA {

template<typename taNumber> class CERadixSortRatingsSubtaskSort : public SRPlat::SRStandardSubtask {
public: // types
  typedef CERadixSortRatingsTask<taNumber> TTask;

private: // variables
  TPqaId *PTR_RESTRICT _pCounters;

private: // methods
  TPqaAmount Flip(TPqaAmount x);
  TPqaAmount Unflip(TPqaAmount y);
  void ZeroCounters();
public: // methods
  using SRPlat::SRStandardSubtask::SRStandardSubtask;
  virtual void Run() override final;
};

} // namespace ProbQA
