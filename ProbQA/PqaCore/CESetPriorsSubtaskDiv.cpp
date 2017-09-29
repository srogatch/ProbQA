// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CESetPriorsSubtaskDiv.h"
#include "../PqaCore/CESetPriorsTask.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template class CESetPriorsSubtaskDiv<SRDoubleNumber>;

template<typename taNumber> CESetPriorsSubtaskDiv<taNumber>::CESetPriorsSubtaskDiv(TTask *pTask)
  : SRStandardSubtask(pTask) { }

//TODO: generalize with CENormPriorsSubtaskDiv
template<> void CESetPriorsSubtaskDiv<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();

  auto *PTR_RESTRICT pMants = SRCast::Ptr<__m256d>(quiz.GetPriorMants());

  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    const __m256d original = SRSimd::Load<false>(pMants + i);
    const __m256d normalized = _mm256_div_pd(original, task._sumPriors._comps);
    SRSimd::Store<false>(pMants + i, original);
  }
  _mm_sfence();
}

} // namespace ProbQA
