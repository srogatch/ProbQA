// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CECreateQuizOperation.h"
#include "../PqaCore/CEUpdatePriorsTask.h"
#include "../PqaCore/CEUpdatePriorsSubtaskMul.h"
#include "../PqaCore/CEQuiz.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/DoubleNumber.h"

using namespace SRPlat;

namespace ProbQA {

template<> void CECreateQuizResume<DoubleNumber>::ApplyAnsweredQuestions(
  CpuEngine<DoubleNumber> *pCe, CEQuiz<DoubleNumber> *pQuiz)
{
  //The input must have been validated
  const EngineDimensions& dims = pCe->GetDims();
  const size_t nVects = SRSimd::VectsFromComps<double>(dims._nTargets);
  const SRThreadPool::TThreadCount nWorkers = pCe->GetWorkers().GetWorkerCount();
  const size_t sizeWithSubtasks = sizeof(CEUpdatePriorsSubtaskMul<DoubleNumber>) * nWorkers;
  SRSmartMPP<CpuEngine<DoubleNumber>::TMemPool, uint8_t> commonBuf(pCe->GetMemPool(),
    /* This assumes that the subtasks end with the buffer end. */ sizeWithSubtasks);
  CEUpdatePriorsTask<DoubleNumber> task(pCe, pQuiz, _nAnswered, _pAQs, CalcVectsInCache());

  {
    SRRWLock<false> rwl(pCe->GetRws());

    pCe->SplitAndRunSubtasksSlim<CEUpdatePriorsSubtaskMul<DoubleNumber>>(task, nVects,
      /* This assumes that the subtasks are at the beginning of the buffer. */ commonBuf.Get(),
      [&](CEUpdatePriorsSubtaskMul<DoubleNumber> *pCurSt, const size_t curStart, const size_t nextStart)
    {
      new (pCurSt) CEUpdatePriorsSubtaskMul<DoubleNumber>(&task, curStart, nextStart);
    });
  }
}

template class CECreateQuizResume<DoubleNumber>;

} // namespace ProbQA
