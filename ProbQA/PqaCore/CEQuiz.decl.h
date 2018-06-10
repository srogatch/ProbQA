// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/BaseCpuEngine.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

//NOTE: destructor is not virtual, so don't try to destruct the derived using a pointer to base.
class CEBaseQuiz {
public: // types
  typedef int64_t TExponent;

protected: // types
  typedef BaseCpuEngine::TMemPool TMemPool;

protected:
  std::vector<AnsweredQuestion> _answers;

private:
  BaseCpuEngine *_pEngine;
  // For each question, the corresponding bit indicates whether it has already been asked in this quiz
  __m256i *_isQAsked;
  // Exponents for the target likelyhoods: x[i] = _pTlhs[i] * pow(2, _pExps[i])
  TExponent *_pTlhExps;

protected: // variables
  TPqaId _activeQuestion = cInvalidPqaId;

protected: // methods
  inline explicit CEBaseQuiz(BaseCpuEngine *pEngine);
  inline ~CEBaseQuiz();
  BaseCpuEngine* GetBaseEngine() const { return _pEngine; }

public: // methods
  TExponent* GetTlhExps() const { return _pTlhExps; }
  __m256i* GetQAsked() const { return _isQAsked; }
  std::vector<AnsweredQuestion>& ModAnswers() { return _answers; }
  const std::vector<AnsweredQuestion>& GetAnswers() const { return _answers; }
  void SetActiveQuestion(TPqaId iQuestion) { _activeQuestion = iQuestion; }
  TPqaId GetActiveQuestion() const { return _activeQuestion; }
};

template<typename taNumber> class CEQuiz : public CEBaseQuiz {
  // For precision and to avoid underflow, mantissas and exponents are stored separately.
  // Priors must be usually normalized, except for short periods of updating them.
  taNumber *_pPriorMants;

public: // methods
  explicit CEQuiz(CpuEngine<taNumber> *pEngine);
  ~CEQuiz();
  taNumber* GetPriorMants() const { return _pPriorMants; }
  CpuEngine<taNumber>* GetEngine() const;
  inline PqaError RecordAnswer(const TPqaId iAnswer);
};

} // namespace ProbQA
