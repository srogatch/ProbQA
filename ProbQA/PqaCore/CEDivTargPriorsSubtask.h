// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEDivTargPriorsSubtask.decl.h"
#include "../PqaCore/CEQuiz.h"

namespace ProbQA {

template<> inline void __vectorcall CEBaseDivTargPriorsSubtask<SRPlat::SRDoubleNumber>::RunInternal(
  const CEQuiz<SRPlat::SRDoubleNumber> &PTR_RESTRICT quiz, const SRPlat::SRNumPack<SRPlat::SRDoubleNumber> sumPriors)
{
  auto *PTR_RESTRICT pMants = SRPlat::SRCast::Ptr<__m256d>(quiz.GetPriorMants());
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    const __m256d original = SRPlat::SRSimd::Load<false>(pMants + i);
    const __m256d normalized = _mm256_div_pd(original, sumPriors._comps);
    SRPlat::SRSimd::Store<false>(pMants + i, original);
  }
  _mm_sfence();
}

template<typename taTask> inline void CEDivTargPriorsSubtask<taTask>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  RunInternal(quiz, task._sumPriors);
}

} // namespace ProbQA
