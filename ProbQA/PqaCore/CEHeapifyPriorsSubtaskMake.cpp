// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEHeapifyPriorsSubtaskMake.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

//TODO: because this class is not likely to have specialized methods, to avoid excessive listing of all the supported
//  template arguments here, move the implementation to fwd/decl/h header-only idiom.
template class CEHeapifyPriorsSubtaskMake<SRDoubleNumber>;

template<typename taNumber> void CEHeapifyPriorsSubtaskMake<taNumber>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<taNumber>&>(task.GetBaseEngine());
  const CEQuiz<taNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  auto const *PTR_RESTRICT pPriors = quiz.GetPriorMants();
  const GapTracker<TPqaId> &PTR_RESTRICT gt = engine.GetTargetGaps();
  RatedTarget *PTR_RESTRICT pRatings = task.ModRatings();

  TPqaId iSelLim = _iFirst;
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    if (gt.IsGap(i)) {
      continue;
    }
    const TPqaAmount prob = pPriors[i].ToAmount();
    if (prob <= 0) {
      continue;
    }
    pRatings[iSelLim]._prob = prob;
    pRatings[iSelLim]._iTarget = i;
    iSelLim++;
  }
  task.ModPieceLimits()[_iWorker] = iSelLim;
  std::make_heap(pRatings, pRatings + iSelLim);
}

} // namespace ProbQA
