// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CERadixSortRatingsSubtaskSort.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/CEQuiz.h"
#include "../PqaCore/CEListTopTargetsAlgorithm.h"

using namespace SRPlat;

namespace ProbQA {

//TODO: because this class is not likely to have specialized methods, to avoid excessive listing of all the supported
//  template arguments here, move the implementation to fwd/decl/h header-only idiom.
template class CERadixSortRatingsSubtaskSort<SRDoubleNumber>;

static_assert(std::is_same<TPqaAmount, double>::value, "The methods below are for double only.");
template<typename taNumber> TPqaAmount CERadixSortRatingsSubtaskSort<taNumber>::Flip(TPqaAmount x) {
  const uint64_t iX = SRCast::U64FromF64(x);
  const uint64_t mask = -int64_t(iX >> 63) | (1ui64 << 63);
  return SRCast::F64FromU64(iX ^ mask);
}

template<typename taNumber> TPqaAmount CERadixSortRatingsSubtaskSort<taNumber>::Unflip(TPqaAmount y) {
  const uint64_t iY = SRCast::U64FromF64(y);
  const uint64_t mask = ((iY >> 63) - 1) | (1ui64 << 63);
  return SRCast::F64FromU64(iY ^ mask);
}

template<typename taNumber> void CERadixSortRatingsSubtaskSort<taNumber>::ZeroCounters() {
  constexpr size_t cnBuckets = CEListTopTargetsAlgorithm<taNumber>::_cnRadixSortBuckets;
  static_assert(sizeof(TPqaId)*cnBuckets % SRSimd::_cNBytes == 0, "Shifted without rounding up.");
  SRUtils::FillZeroVects<true>(SRCast::Ptr<__m256i>(_pCounters), (sizeof(TPqaId)*cnBuckets) >> SRSimd::_cLogNBytes);
}

template<typename taNumber> void CERadixSortRatingsSubtaskSort<taNumber>::Run() {
  constexpr size_t cnBuckets = CEListTopTargetsAlgorithm<taNumber>::_cnRadixSortBuckets;
  static_assert(cnBuckets == 256, "Relied on in masks below.");
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<taNumber>&>(task.GetBaseEngine());
  const CEQuiz<taNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  auto const *const PTR_RESTRICT pPriors = quiz.GetPriorMants();
  const GapTracker<TPqaId> &PTR_RESTRICT gt = engine.GetTargetGaps();
  RatedTarget *PTR_RESTRICT pRatings = task.ModRatings() + _iWorker;
  RatedTarget *PTR_RESTRICT pTempRatings = task.ModTempRatings() + _iWorker;
  _pCounters = task.ModCounters() + _iWorker * cnBuckets;
  TPqaId *const PTR_RESTRICT pOffsets = task.ModOffsets() + _iWorker * cnBuckets;

  // Radix sort based on: http://stereopsis.com/radix.html , http://www.codercorner.com/RadixSortRevisited.htm

  ZeroCounters();
  TPqaId iSelLim = _iFirst;

  // pass 0 with flipping
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    if (gt.IsGap(i)) {
      continue;
    }
    const TPqaAmount prob = pPriors[i].ToAmount();
    if (prob <= 0) {
      continue;
    }
    const TPqaAmount flipped = Flip(prob);
    pRatings[iSelLim]._iTarget = i;
    pRatings[iSelLim]._prob = flipped;
    iSelLim++;
    _pCounters[SRCast::U64FromF64(flipped) & 0xff]++;
  }

  // passes 1 to 7 without flipping or unflipping
  for (uint8_t i = 1; i<=7; i++) {
    pOffsets[0] = 0;
    for (size_t j = 1; j < cnBuckets; j++) {
      pOffsets[j] = pOffsets[j - 1] + _pCounters[j - 1];
    }
    ZeroCounters();
    for (TPqaId j = _iFirst; j < iSelLim; j++) {
      const uint64_t iProb = SRCast::U64FromF64(pRatings[j]._prob);
      pTempRatings[pOffsets[(iProb >> ((i - 1) << 3)) & 0xff]] = pRatings[j];
      _pCounters[(iProb >> (i << 3)) & 0xff]++;
    }
    std::swap(pRatings, pTempRatings);
  }

  // pass 9 with unflipping
  pOffsets[0] = 0;
  for (size_t j = 1; j < cnBuckets; j++) {
    pOffsets[j] = pOffsets[j - 1] + _pCounters[j - 1];
  }
  for (TPqaId j = _iFirst; j < iSelLim; j++) {
    const uint64_t iProb = SRCast::U64FromF64(pRatings[j]._prob);
    RatedTarget &dest = pTempRatings[pOffsets[(iProb >> 56) & 0xff]];
    dest._iTarget = pRatings[j]._iTarget;
    dest._prob = Unflip(pRatings[j]._prob);
  }

  // Put sentinel item
  pTempRatings[iSelLim]._prob = 0;
}

} // namespace ProbQA
