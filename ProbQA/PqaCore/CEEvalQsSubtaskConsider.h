// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEEvalQsTask.fwd.h"

namespace ProbQA {

template<typename taNumber> class CEEvalQsSubtaskConsider : public SRPlat::SRStandardSubtask {
public: // types
  typedef CEEvalQsTask<taNumber> TTask;

public: // constants
  static const size_t _cAccumVectSize;

public: // methods
  using SRPlat::SRStandardSubtask::SRStandardSubtask;
  virtual void Run() override final;
};

} // namespace ProbQA
