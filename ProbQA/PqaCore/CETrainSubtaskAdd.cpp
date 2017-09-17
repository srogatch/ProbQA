// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CETrainSubtaskAdd.h"
#include "../PqaCore/CETrainTask.h"
#include "../PqaCore/CpuEngine.h"

using namespace SRPlat;

namespace ProbQA {

template class CETrainSubtaskAdd<SRDoubleNumber>;

template<> void CETrainSubtaskAdd<SRDoubleNumber>::Run() {
  auto& cTask = static_cast<const TTask&>(*GetTask()); // enable optimizations with const
  auto& engine = static_cast<CpuEngine<SRDoubleNumber>&>(cTask.GetBaseEngine());
  TPqaId iLast = cTask._last[_iWorker];
  if (iLast == cInvalidPqaId) {
    return;
  }
  const TPqaId *const cPrev = cTask._prev;
  // Enable optimizations with const
  const __m256d& fullAddend = cTask._numSpec._fullAddend;
  const __m256d& collAddend = cTask._numSpec._collAddend;
  do {
    const AnsweredQuestion& aqFirst = cTask._pAQs[iLast];
    iLast = cPrev[iLast];
    if (iLast == cInvalidPqaId) {
      // Use SSE2 instead of AVX here to supposedly reduce the load on the CPU core (better hyperthreading).
      __m128d sum = _mm_set_pd(
        engine.GetD(aqFirst._iQuestion, cTask._iTarget).GetValue(),
        engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, cTask._iTarget).GetValue());
      sum = _mm_add_pd(sum, _mm256_castpd256_pd128(fullAddend));
      engine.ModA(aqFirst._iQuestion, aqFirst._iAnswer, cTask._iTarget).SetValue(sum.m128d_f64[0]);
      engine.ModD(aqFirst._iQuestion, cTask._iTarget).SetValue(sum.m128d_f64[1]);
      return;
    }
    const AnsweredQuestion& aqSecond = cTask._pAQs[iLast];
    if (aqFirst._iQuestion == aqSecond._iQuestion) {
      // Vectorize 3 additions, with twice the amount in element #1
      __m256d sum = _mm256_set_pd(0,
        engine.GetA(aqSecond._iQuestion, aqSecond._iAnswer, cTask._iTarget).GetValue(),
        engine.GetD(aqFirst._iQuestion, cTask._iTarget).GetValue(),
        engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, cTask._iTarget).GetValue());
      sum = _mm256_add_pd(sum, collAddend);
      engine.ModA(aqFirst._iQuestion, aqFirst._iAnswer, cTask._iTarget).SetValue(sum.m256d_f64[0]);
      engine.ModD(aqFirst._iQuestion, cTask._iTarget).SetValue(sum.m256d_f64[1]);
      engine.ModA(aqSecond._iQuestion, aqSecond._iAnswer, cTask._iTarget).SetValue(sum.m256d_f64[2]);
    }
    else {
      // Finally we can vectorize all the 4 additions
      __m256d sum = _mm256_set_pd(
        engine.GetD(aqSecond._iQuestion, cTask._iTarget).GetValue(),
        engine.GetA(aqSecond._iQuestion, aqSecond._iAnswer, cTask._iTarget).GetValue(),
        engine.GetD(aqFirst._iQuestion, cTask._iTarget).GetValue(),
        engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, cTask._iTarget).GetValue());
      sum = _mm256_add_pd(sum, fullAddend);
      engine.ModA(aqFirst._iQuestion, aqFirst._iAnswer, cTask._iTarget).SetValue(sum.m256d_f64[0]);
      engine.ModD(aqFirst._iQuestion, cTask._iTarget).SetValue(sum.m256d_f64[1]);
      engine.ModA(aqSecond._iQuestion, aqSecond._iAnswer, cTask._iTarget).SetValue(sum.m256d_f64[2]);
      engine.ModD(aqSecond._iQuestion, cTask._iTarget).SetValue(sum.m256d_f64[3]);
    }
    iLast = cPrev[iLast];
  } while (iLast != cInvalidPqaId);
}

} // namespace ProbQA
