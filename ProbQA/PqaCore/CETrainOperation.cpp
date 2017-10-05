// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CETrainOperation.h"
#include "../PqaCore/CpuEngine.h"

using namespace SRPlat;

namespace ProbQA {

template class CETrainOperation<SRDoubleNumber>;

namespace {

// Collision addend: amount is added twice to _mD[iQuestion][iTarget] .
__m256d MakeCollisionAddend(__m256d fullAddend) {
  fullAddend.m256d_f64[1] += fullAddend.m256d_f64[1];
  return fullAddend;
}

} // anonymous namespace

template<> void CETrainOperation<SRDoubleNumber>::Perform() {
  const double dAmount = SRCast::ToDouble(_amount);
  const __m256d fullAddend = _mm256_set1_pd(dAmount);
  const __m256d collAddend = MakeCollisionAddend(fullAddend);

  SRRWLock<true> rwl(_engine.GetRws());
  TPqaId i = 0;
  for (const TPqaId iEn = _nQuestions-1; i < iEn ; i+=2) {
    const AnsweredQuestion& aqFirst = _pAQs[i];
    const AnsweredQuestion& aqSecond = _pAQs[i + 1];
    if (aqFirst._iQuestion == aqSecond._iQuestion) {
      // Vectorize 3 additions, with twice the amount in element #1
      __m256d sum = _mm256_set_pd(0,
        _engine.GetA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).GetValue(),
        _engine.GetD(aqFirst._iQuestion, _iTarget).GetValue(),
        _engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).GetValue());
      sum = _mm256_add_pd(sum, collAddend);
      _engine.ModA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).SetValue(sum.m256d_f64[0]);
      _engine.ModD(aqFirst._iQuestion, _iTarget).SetValue(sum.m256d_f64[1]);
      _engine.ModA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).SetValue(sum.m256d_f64[2]);
    }
    else {
      // We can vectorize all the 4 additions
      __m256d sum = _mm256_set_pd(
        _engine.GetD(aqSecond._iQuestion, _iTarget).GetValue(),
        _engine.GetA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).GetValue(),
        _engine.GetD(aqFirst._iQuestion, _iTarget).GetValue(),
        _engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).GetValue());
      sum = _mm256_add_pd(sum, fullAddend);
      _engine.ModA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).SetValue(sum.m256d_f64[0]);
      _engine.ModD(aqFirst._iQuestion, _iTarget).SetValue(sum.m256d_f64[1]);
      _engine.ModA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).SetValue(sum.m256d_f64[2]);
      _engine.ModD(aqSecond._iQuestion, _iTarget).SetValue(sum.m256d_f64[3]);
    }
  }
  assert(_nQuestions - 1 <= i && i <= _nQuestions);
  if (i + 1 == _nQuestions) {
    const AnsweredQuestion& aqFirst = _pAQs[i];
    // Use SSE2 instead of AVX here to supposedly reduce the load on the CPU core (better hyperthreading).
    __m128d sum = _mm_set_pd(
      _engine.GetD(aqFirst._iQuestion, _iTarget).GetValue(),
      _engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).GetValue());
    sum = _mm_add_pd(sum, _mm256_castpd256_pd128(fullAddend));
    _engine.ModA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).SetValue(sum.m128d_f64[0]);
    _engine.ModD(aqFirst._iQuestion, _iTarget).SetValue(sum.m128d_f64[1]);
  }
  _engine.ModB(_iTarget).ModValue() += dAmount;
}

} // namespace ProbQA
