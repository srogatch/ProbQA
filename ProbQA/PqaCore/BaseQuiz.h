// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

class BaseQuiz {
protected: // types
  typedef BaseEngine::TMemPool TMemPool;

protected: // variables
  std::vector<AnsweredQuestion> _answers;
  TPqaId _activeQuestion = cInvalidPqaId;

private: // variables
  BaseEngine *_pEngine;
  // For each question, the corresponding bit indicates whether it has already been asked in this quiz
  __m256i *_isQAsked;

protected: // methods
  BaseEngine * GetBaseEngine() const { return _pEngine; }
  virtual ~BaseQuiz();

public: // methods
  __m256i* GetQAsked() const { return _isQAsked; }
};

} // namespace ProbQA
