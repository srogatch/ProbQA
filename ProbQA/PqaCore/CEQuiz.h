// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CpuEngine;

template<typename taNumber> class CEQuiz {
public:
  static const size_t _cLogSimdBits = 8;
  typedef int64_t TExponent;

private: // variables
  std::vector<AnsweredQuestion> _answers;
  // Target LikeliHood mantissas: for better performance, they are usually NOT normalized to probabilities by dividing
  //   by their sum. For precision and to avoid underflow, mantissas and exponents are stored separately.
  taNumber *_pTlhMants;
  // Exponents for the target likelyhoods: x[i] = _pTlhs[i] * pow(2, _pExps[i])
  TExponent *_pTlhExps;
  // For each question, the corresponding bit indicates whether it has already been asked in this quiz
  __m256i *_isQAsked;
  CpuEngine<taNumber> *_pEngine;
  TPqaId _activeQuestion = cInvalidPqaId;
public:
  explicit CEQuiz(CpuEngine<taNumber> *pEngine);
  ~CEQuiz();
  taNumber* GetTlhMants() { return _pTlhMants; }
  TExponent* GetTlhExps() { return _pTlhExps; }
  __m256i *GetQAsked() { return _isQAsked; }
  CpuEngine<taNumber>* GetEngine() { return _pEngine; }
};

} // namespace ProbQA
