// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CERecordAnswerTask.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEBaseTask.h"

namespace ProbQA {

template<typename taNumber> class CERecordAnswerTask : public CEBaseTask {
public: // types
  typedef taNumber TNumber;

private: // variables
  const AnsweredQuestion _aq;
  CEQuiz<taNumber> *_pQuiz;
  SRPlat::SRBucketSummatorPar<taNumber> *const _pBsp;
public: // variables
  SRPlat::SRNumPack<taNumber> _sumPriors;

public:
  explicit CERecordAnswerTask(CpuEngine<taNumber> &engine, CEQuiz<taNumber> &quiz, const AnsweredQuestion& aq,
    SRPlat::SRBucketSummatorPar<taNumber> &bsp) : CEBaseTask(engine), _pQuiz(&quiz), _aq(aq), _pBsp(&bsp) { }

  const AnsweredQuestion& GetAQ() const { return _aq; }
  CEQuiz<taNumber>& GetQuiz() const { return *_pQuiz; }
  SRPlat::SRBucketSummatorPar<taNumber>& GetBSP() const { return *_pBsp; }
};

} // namespace ProbQA
