#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CpuEngine;

template<typename taNumber> class CEQuiz {
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
};

} // namespace ProbQA
