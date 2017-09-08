// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CECreateQuizOperation.h"
#include "../PqaCore/CEUpdatePriorsTask.h"
#include "../PqaCore/CEUpdatePriorsSubtaskMul.h"
#include "../PqaCore/CEQuiz.h"
#include "../PqaCore/CpuEngine.h"

using namespace SRPlat;

namespace ProbQA {

template<> void CECreateQuizResume<SRDoubleNumber>::UpdateLikelihoods(BaseCpuEngine &baseCe, CEBaseQuiz &baseQuiz)
{
  auto &engine = static_cast<CpuEngine<SRDoubleNumber>&>(baseCe);
  auto &quiz = static_cast<CEQuiz<SRDoubleNumber>&>(baseQuiz);
  //The input must have been validated
  const EngineDimensions& dims = engine.GetDims();
  const size_t nVects = SRSimd::VectsFromComps<double>(dims._nTargets);
  const SRThreadCount nWorkers = engine.GetWorkers().GetWorkerCount();
  const size_t sizeWithSubtasks = sizeof(CEUpdatePriorsSubtaskMul<SRDoubleNumber>) * nWorkers;
  SRSmartMPP<uint8_t> commonBuf(engine.GetMemPool(),
    /* This assumes that the subtasks end with the buffer end. */ sizeWithSubtasks);
  CEUpdatePriorsTask<SRDoubleNumber> task(engine, quiz, _nAnswered, _pAQs, CalcVectsInCache());

  {
    SRRWLock<false> rwl(engine.GetRws());

    engine.SplitAndRunSubtasksSlim<CEUpdatePriorsSubtaskMul<SRDoubleNumber>>(task, nVects,
      /* This assumes that the subtasks are at the beginning of the buffer. */ commonBuf.Get(),
      [&](CEUpdatePriorsSubtaskMul<SRDoubleNumber> *pCurSt, const size_t curStart, const size_t nextStart)
    {
      new (pCurSt) CEUpdatePriorsSubtaskMul<SRDoubleNumber>(&task, curStart, nextStart);
    });
  }
}

template class CECreateQuizResume<SRDoubleNumber>;

} // namespace ProbQA
