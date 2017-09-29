// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CESetPriorsTask.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEBaseTask.h"

namespace ProbQA {

template<typename taNumber> class CESetPriorsTask : public CEBaseTask {
  const CEQuiz<taNumber> *const _pQuiz;
  SRPlat::SRBucketSummatorPar<taNumber> *const _pBsp;
public: // variables
  SRPlat::SRNumPack<taNumber> _sumPriors;

public:
  CESetPriorsTask(CpuEngine<taNumber> &engine, CEQuiz<taNumber> &quiz, SRPlat::SRBucketSummatorPar<taNumber> &bsp)
    : CEBaseTask(engine), _pQuiz(&quiz), _pBsp(&bsp) { }

  const CEQuiz<taNumber>& GetQuiz() const { return *_pQuiz; }
  SRPlat::SRBucketSummatorPar<taNumber>& GetBSP() const { return *_pBsp; }
};

} // namespace ProbQA
