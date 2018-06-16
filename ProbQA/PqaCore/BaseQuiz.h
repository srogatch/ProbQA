// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseEngine.h"

namespace ProbQA {

class BaseEngine;

class BaseQuiz {
protected: // types
  typedef BaseEngine::TMemPool TMemPool;

protected: // variables
  std::vector<AnsweredQuestion> _answers;
  TPqaId _activeQuestion = cInvalidPqaId;
  BaseEngine *_pEngine;

private: // variables
  // For each question, the corresponding bit indicates whether it has already been asked in this quiz
  __m256i *_isQAsked;

protected: // methods
  BaseEngine * GetBaseEngine() const { return _pEngine; }
  virtual ~BaseQuiz();
  explicit BaseQuiz(BaseEngine *pEngine);

public: // methods
  __m256i* GetQAsked() const { return _isQAsked; }
  virtual PqaError RecordAnswer(const TPqaId iAnswer) = 0;
  std::vector<AnsweredQuestion>& ModAnswers() { return _answers; }
  const std::vector<AnsweredQuestion>& GetAnswers() const { return _answers; }
  void SetActiveQuestion(TPqaId iQuestion) { _activeQuestion = iQuestion; }
  TPqaId GetActiveQuestion() const { return _activeQuestion; }
};

} // namespace ProbQA
