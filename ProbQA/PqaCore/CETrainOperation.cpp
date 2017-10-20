// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CETrainOperation.h"
#include "../PqaCore/CpuEngine.h"

using namespace SRPlat;

namespace ProbQA {

template class CETrainOperation<SRDoubleNumber>;

void CETrainOperation<SRDoubleNumber>::ProcessOne(const AnsweredQuestion& aq, const double twoB, const double bSquare) {
  // Use SSE2 instead of AVX here to supposedly reduce the load on the CPU core (better hyperthreading).
  const double aSquare = _engine.GetA(aq._iQuestion, aq._iAnswer, _iTarget).GetValue();
  const double a = std::sqrt(aSquare);
  const __m128d sseAddend = _mm_set1_pd(a * twoB + bSquare);
  __m128d sum = _mm_set_pd(
    _engine.GetD(aq._iQuestion, _iTarget).GetValue(),
    _engine.GetA(aq._iQuestion, aq._iAnswer, _iTarget).GetValue());
  sum = _mm_add_pd(sum, sseAddend);
  _engine.ModA(aq._iQuestion, aq._iAnswer, _iTarget).SetValue(sum.m128d_f64[0]);
  _engine.ModD(aq._iQuestion, _iTarget).SetValue(sum.m128d_f64[1]);
}

template<> void CETrainOperation<SRDoubleNumber>::Perform() {
  const double dAmount = SRCast::ToDouble(_amount);
  // (a+b)**2 = a**2 + 2*a*b + b**2
  const __m256d b = _mm256_set1_pd(dAmount);
  const __m256d twoB = _mm256_add_pd(b, b);
  const __m256d bSquare = _mm256_mul_pd(b, b);

  SRRWLock<true> rwl(_engine.GetRws());
  TPqaId i = 0;
  for (const TPqaId iEn = _nQuestions-1; i < iEn ; i+=2) {
    const AnsweredQuestion& aqFirst = _pAQs[i];
    const AnsweredQuestion& aqSecond = _pAQs[i + 1];
    //TODO: extend to 4 questions at once
    if (aqFirst._iQuestion == aqSecond._iQuestion) {
      if (aqFirst._iAnswer == aqSecond._iAnswer) {
        const double fourB = twoB.m256d_f64[0] + twoB.m256d_f64[0];
        const double twoBSquare = bSquare.m256d_f64[0] * 4;
        ProcessOne(aqFirst, fourB, twoBSquare);
      }
      else {
        // Vectorize 3 additions, with twice the amount in element #2
        const __m128d aSquare = _mm_set_pd(_engine.GetA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).GetValue(),
          _engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).GetValue());
        const __m128d a = _mm_sqrt_pd(aSquare);
        const __m128d ab2 = _mm_mul_pd(a, _mm256_castpd256_pd128(twoB));
        const __m128d sseAddend = _mm_add_pd(ab2, _mm256_castpd256_pd128(bSquare));
        const __m256d avxAddend = _mm256_set_m128d(_mm_set1_pd(sseAddend.m128d_f64[0]+ sseAddend.m128d_f64[0]),
          sseAddend);
        __m256d sum = _mm256_set_pd(0,
          _engine.GetD(aqFirst._iQuestion, _iTarget).GetValue(),
          _engine.GetA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).GetValue(),
          _engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).GetValue());
        sum = _mm256_add_pd(sum, avxAddend);
        _engine.ModA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).SetValue(sum.m256d_f64[0]);
        _engine.ModA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).SetValue(sum.m256d_f64[1]);
        _engine.ModD(aqFirst._iQuestion, _iTarget).SetValue(sum.m256d_f64[2]);
      }
    }
    else {
      // We can vectorize all the 4 additions
      const __m128d aSquare = _mm_set_pd(_engine.GetA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).GetValue(),
        _engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).GetValue());
      const __m128d a = _mm_sqrt_pd(aSquare);
      const __m128d ab2 = _mm_mul_pd(a, _mm256_castpd256_pd128(twoB));
      const __m128d sseAddend = _mm_add_pd(ab2, _mm256_castpd256_pd128(bSquare));
      const __m256d avxAddend = _mm256_castsi256_pd(_mm256_broadcastsi128_si256(_mm_castpd_si128(sseAddend)));

      __m256d sum = _mm256_set_pd(
        _engine.GetD(aqSecond._iQuestion, _iTarget).GetValue(),
        _engine.GetD(aqFirst._iQuestion, _iTarget).GetValue(),
        _engine.GetA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).GetValue(),
        _engine.GetA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).GetValue());

      sum = _mm256_add_pd(sum, avxAddend);

      _engine.ModA(aqFirst._iQuestion, aqFirst._iAnswer, _iTarget).SetValue(sum.m256d_f64[0]);
      _engine.ModA(aqSecond._iQuestion, aqSecond._iAnswer, _iTarget).SetValue(sum.m256d_f64[1]);
      _engine.ModD(aqFirst._iQuestion, _iTarget).SetValue(sum.m256d_f64[2]);
      _engine.ModD(aqSecond._iQuestion, _iTarget).SetValue(sum.m256d_f64[3]);
    }
  }
  assert(_nQuestions - 1 <= i && i <= _nQuestions);
  if (i + 1 == _nQuestions) {
    ProcessOne(_pAQs[i], twoB.m256d_f64[0], bSquare.m256d_f64[0]);
  }
  _engine.ModB(_iTarget).ModValue() += dAmount;
}

} // namespace ProbQA
