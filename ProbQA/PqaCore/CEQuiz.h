// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CpuEngine;

template<typename taNumber> class CEQuiz {
  static const size_t _cLogSimdBits = 8;
private: // variables
  std::vector<AnsweredQuestion> _answers;
  // Probabilities of targets
  taNumber *_pTargProbs;
  // For each question, the corresponding bit indicates whether it has already been asked in this quiz
  __m256i *_isQAsked;
  CpuEngine<taNumber> *_pEngine;
  TPqaId _activeQuestion = cInvalidPqaId;
public:
  explicit CEQuiz(CpuEngine<taNumber> *pEngine);
  ~CEQuiz();
  taNumber* GetTargProbs() { return _pTargProbs; }
  CpuEngine<taNumber>* GetEngine() { return _pEngine; }
};

} // namespace ProbQA
