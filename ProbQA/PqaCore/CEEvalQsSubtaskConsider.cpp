// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEEvalQsSubtaskConsider.h"
#include "../PqaCore/CEEvalQsTask.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template class CEEvalQsSubtaskConsider<SRDoubleNumber>;

template<typename taNumber> CEEvalQsSubtaskConsider<taNumber>::CEEvalQsSubtaskConsider(TTask *pTask)
  : SRStandardSubtask(pTask) { }

template<> void CEEvalQsSubtaskConsider<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  const __m256d *pPriors = SRCast::CPtr<__m256d>(quiz.GetTlhMants());
  SRBucketSummator<SRDoubleNumber> bs(1, task._pBSes
    + SRBucketSummator<SRDoubleNumber>::GetMemoryRequirementBytes(1) * _iWorker);

  double prevRunLength = 0;
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    if (engine.GetQuestionGaps().IsGap(i) || SRBitHelper::Test(quiz.GetQAsked(), i)) {
      // Set 0 probability to this question
      task._pRunLength[i].SetValue(prevRunLength);
      continue;
    }
  }
  //TODO: implement
}

} // namespace ProbQA
