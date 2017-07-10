// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaErrors.h"

namespace ProbQA {

template<typename taNumber> class CpuEngine;
template<typename taNumber> class CEQuiz;

template<typename taNumber> class CECreateQuizOpBase {
public: // constants
  static const bool _cCachePriors; // must be unresolved

public: // variables
  PqaError& _err;
  CEQuiz<taNumber> *_pQuiz;

public: // methods
  explicit CECreateQuizOpBase(PqaError& err) : _err(err) { }
  void MaybeUpdatePriorsWithAnsweredQuestions(CpuEngine<taNumber> *pEngine);
};

template<typename taNumber> class CECreateQuizStart : public CECreateQuizOpBase<taNumber> {
public: // constants
  static const bool _cCachePriors = false;

public: //methods
  explicit CECreateQuizStart(PqaError& err) : CECreateQuizOpBase(err) { }
  void MaybeUpdatePriorsWithAnsweredQuestions(CpuEngine<taNumber> *pEngine) { (void)pEngine; }
};

template<typename taNumber> class CECreateQuizResume : public CECreateQuizOpBase<taNumber> {
public: // constants
  static const bool _cCachePriors = true;

public: // variables
  const TPqaId _nQuestions;
  const AnsweredQuestion* const _pAQs;

public: //methods
  explicit CECreateQuizResume(PqaError& err, const TPqaId nQuestions, const AnsweredQuestion* const pAQs)
    : CECreateQuizOpBase(err), _nQuestions(nQuestions), _pAQs(pAQs)
  { }
  void MaybeUpdatePriorsWithAnsweredQuestions(CpuEngine<taNumber> *pEngine);
};

#include "../PqaCore/CpuEngine.h"

template<typename taNumber> inline void CECreateQuizResume<taNumber>::MaybeUpdatePriorsWithAnsweredQuestions(
  CpuEngine<taNumber> *pEngine)
{
  pEngine->UpdatePriorsWithAnsweredQuestions(*this);
}

} // namespace ProbQA