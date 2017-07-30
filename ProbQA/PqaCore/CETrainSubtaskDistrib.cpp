// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CETrainSubtaskDistrib.h"
#include "../PqaCore/CETrainTask.h"
#include "../PqaCore/DoubleNumber.h"
#include "../PqaCore/CpuEngine.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> CETrainSubtaskDistrib<taNumber>::CETrainSubtaskDistrib(CETrainTask<taNumber> *pTask,
  const AnsweredQuestion *pFirst, const AnsweredQuestion *pLim)
  : SRBaseSubtask(pTask), _pFirst(pFirst), _pLim(pLim)
{
}

template<typename taNumber> void CETrainSubtaskDistrib<taNumber>::Run() {
  auto &task = static_cast<CETrainTask<taNumber>&>(*GetTask());
  auto& engine = static_cast<const CpuEngine<taNumber>&>(*task.GetEngine());
  const EngineDimensions& dims = engine.GetDims();
  const SRThreadPool::TThreadCount nWorkers = task.GetWorkerCount();
  for (const AnsweredQuestion *pAQ = _pFirst, *pEn = _pLim; pAQ < pEn; pAQ++) {
    // Check ranges
    const TPqaId iQuestion = pAQ->_iQuestion;
    if (iQuestion < 0 || iQuestion >= dims._nQuestions) {
      const TPqaId nKB = dims._nQuestions;
      task.AddError(PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iQuestion, 0, nKB - 1),
        SRString::MakeUnowned("Question index is not in KB range.")));
      return;
    }
    if (engine.GetQuestionGaps().IsGap(iQuestion)) {
      task.AddError(PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iQuestion), SRString::MakeUnowned(
        "Question index is not in KB (but rather at a gap).")));
      return;
    }
    const TPqaId iAnswer = pAQ->_iAnswer;
    if (iAnswer < 0 || iAnswer >= dims._nAnswers) {
      const TPqaId nKB = dims._nAnswers;
      task.AddError(PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iAnswer, 0, nKB - 1),
        SRString::MakeUnowned("Answer index is not in KB range.")));
      return;
    }
    // Sort questions into buckets so that workers in the next phase do not race for data.
    const TPqaId iBucket = iQuestion % nWorkers;
    const TPqaId iPrev = task._iPrev.fetch_add(1);
    TPqaId expected = task._last[iBucket].load(std::memory_order_acquire);
    while (!task._last[iBucket].compare_exchange_weak(expected, iPrev, std::memory_order_acq_rel,
      std::memory_order_acquire));
    task._prev[iPrev] = expected;
  }
}

template class CETrainSubtaskDistrib<DoubleNumber>;

} // namespace ProbQA
