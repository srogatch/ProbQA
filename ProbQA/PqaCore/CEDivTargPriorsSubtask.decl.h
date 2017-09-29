// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEDivTargPriorsSubtask.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"

namespace ProbQA {

template<typename taNumber> class CEBaseDivTargPriorsSubtask : public SRPlat::SRStandardSubtask {
public:
  inline void __vectorcall RunInternal(const CEQuiz<taNumber> &PTR_RESTRICT quiz,
    const SRPlat::SRNumPack<taNumber> sumPriors);
  using SRPlat::SRStandardSubtask::SRStandardSubtask;
};

template<typename taTask> class CEDivTargPriorsSubtask : public CEBaseDivTargPriorsSubtask<typename taTask::TNumber> {
public: // types
  typedef taTask TTask;

public: // methods
  using CEBaseDivTargPriorsSubtask<typename taTask::TNumber>::CEBaseDivTargPriorsSubtask;
  inline virtual void Run() override final;
};

} // namespace ProbQA
