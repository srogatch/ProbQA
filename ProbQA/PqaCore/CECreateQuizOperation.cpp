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
  //TODO: validate the input (answered questions)

  //// Sequential code (single-threaded) for reference
  //NOTE: it may be better to iterate by targets first instead, so to apply all multiplications for the first
  //  target and then move on to the next target. This involves 1 unsequential memory access per answered question
  //  application, while if we iterate first by questions, each question application involves 2 memory accesses: load
  //  and store.
  //const TPqaId nTargets = _dims._nTargets;
  //taNumber *pTargProb = resumeOp._pQuiz->GetTargProbs();
  //TPqaId i = 0;
  //for (; i + 1 < resumeOp._nQuestions; i++) {
  //  const AnsweredQuestion& aq = resumeOp._pAQs[i];
  //  for (TPqaId j = 0; j < nTargets; j++) {
  //    // Multiplier compensation is less robust than summation of logarithms, but it's substantially faster and is
  //    //   supported by AVX2. The idea is to make the multipliers equal to 1 in the average case p[j]=1/M, where M is
  //    //   the number of targets.
  // //FIXME: this will blow to infinity the top most likely targets, making them all equal, which is highly undesirable
  //    pTargProb[j] *= (nTargets * _sA[aq._iAnswer][aq._iQuestion][j] / _mD[aq._iQuestion][j]);
  //  }
  //}
  //taNumber sum(0); //TODO: instead, sort then sum
  //const AnsweredQuestion& aq = resumeOp._pAQs[i];
  //for (TPqaId j = 0; j < nTargets; j++) {
  //  taNumber product = pTargProb[j] * (nTargets * _sA[aq._iAnswer][aq._iQuestion][j] / _mD[aq._iQuestion][j]);
  //  pTargProb[j] = product;
  //  sum += product; //TODO: assign to a bucket instead
  //}
  //for (TPqaId j = 0; j < nTargets; j++) {
  //  pTargProb[j] /= sum;
  //}
  const EngineDimensions& dims = pCe->GetDims();
  const size_t nVects = SRSimd::VectsFromComps<double>(dims._nTargets);
  const SRThreadPool::TThreadCount nWorkers = pCe->GetWorkers().GetWorkerCount();
  const size_t sizeWithSubtasks = sizeof(CEUpdatePriorsSubtaskMul<DoubleNumber>) * nWorkers;
  SRSmartMPP<CpuEngine<DoubleNumber>::TMemPool, uint8_t> commonBuf(pCe->GetMemPool(),
    /* This assumes that the subtasks end with the buffer end. */ sizeWithSubtasks);
  CEUpdatePriorsTask<DoubleNumber> task(pCe, pQuiz, _nAnswered, _pAQs, CalcVectsInCache());

  pCe->SplitAndRunSubtasksSlim<CEUpdatePriorsSubtaskMul<DoubleNumber>>(task, nVects,
    /* This assumes that the subtasks are at the beginning of the buffer. */ commonBuf.Get(),
    [&](CEUpdatePriorsSubtaskMul<DoubleNumber> *pCurSt, const size_t curStart, const size_t nextStart)
  {
    new (pCurSt) CEUpdatePriorsSubtaskMul<DoubleNumber>(&task, curStart, nextStart);
  });

  //TODO: finally apply _nAnswered and _pAQs to pQuiz->_isQAsked, pQuiz->_answers and maybe also update
  //  pQuiz->_activeQuestion .
}

template class CECreateQuizResume<DoubleNumber>;

} // namespace ProbQA
