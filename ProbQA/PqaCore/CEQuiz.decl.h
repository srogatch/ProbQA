// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/BaseCpuEngine.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

//NOTE: destructor is not virtual, so don't try to destruct the derived using a pointer to base.
class CEBaseQuiz {
public: // types
  typedef int64_t TExponent;

protected: // types
  typedef BaseCpuEngine::TMemPool TMemPool;

private: // variables
  SRPlat::SRFastRandom _fr;
  std::vector<AnsweredQuestion> _answers;
  BaseCpuEngine *_pEngine;
  // For each question, the corresponding bit indicates whether it has already been asked in this quiz
  __m256i *_isQAsked;
  // Exponents for the target likelyhoods: x[i] = _pTlhs[i] * pow(2, _pExps[i])
  TExponent *_pTlhExps;
  TPqaId _activeQuestion = cInvalidPqaId;

protected: // methods
  explicit CEBaseQuiz(BaseCpuEngine *pEngine);
  ~CEBaseQuiz();
  BaseCpuEngine* GetBaseEngine() const { return _pEngine; }

public: // methods
  TExponent* GetTlhExps() const { return _pTlhExps; }
  __m256i* GetQAsked() const { return _isQAsked; }
  std::vector<AnsweredQuestion>& ModAnswers() { return _answers; }
  SRPlat::SRFastRandom& Random() { return _fr; }
};

template<typename taNumber> class CEQuiz : public CEBaseQuiz {
  // Target LikeliHood mantissas: for better performance, they are usually NOT normalized to probabilities by dividing
  //   by their sum. For precision and to avoid underflow, mantissas and exponents are stored separately.
  taNumber *_pTlhMants;

public: // methods
  explicit CEQuiz(CpuEngine<taNumber> *pEngine);
  ~CEQuiz();
  taNumber* GetTlhMants() const { return _pTlhMants; }
  CpuEngine<taNumber>* GetEngine() const;
};

} // namespace ProbQA
