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

template class CECreateQuizResume<SRDoubleNumber>;

template<> void CECreateQuizResume<SRDoubleNumber>::UpdateLikelihoods(BaseCpuEngine &baseCe, CEBaseQuiz &baseQuiz)
{
  auto &PTR_RESTRICT engine = static_cast<CpuEngine<SRDoubleNumber>&>(baseCe);
  auto &PTR_RESTRICT quiz = static_cast<CEQuiz<SRDoubleNumber>&>(baseQuiz);
  //The input must have been validated
  const EngineDimensions& dims = engine.GetDims();
  const size_t nVects = SRSimd::VectsFromComps<double>(dims._nTargets);
  const SRThreadCount nWorkers = engine.GetWorkers().GetWorkerCount();
  constexpr size_t subtasksOffs = 0;
  const size_t totalBytes = subtasksOffs + nWorkers * SRMaxSizeof<CEUpdatePriorsSubtaskMul<SRDoubleNumber>>::value;
  SRSmartMPP<uint8_t> commonBuf(engine.GetMemPool(), totalBytes);
  CEUpdatePriorsTask<SRDoubleNumber> task(engine, quiz, _nAnswered, _pAQs, CalcVectsInCache());
  SRPoolRunner pr(engine.GetWorkers(), commonBuf.Get() + subtasksOffs);

  {
    SRRWLock<false> rwl(engine.GetRws());
    pr.SplitAndRunSubtasks<CEUpdatePriorsSubtaskMul<SRDoubleNumber>>(task, nVects);
  }
}

} // namespace ProbQA
