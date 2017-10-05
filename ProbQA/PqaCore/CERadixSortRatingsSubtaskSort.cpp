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

namespace {
  static_assert(sizeof(RatedTarget) == sizeof(__m128i), "For SSE streaming below.");
  union SseRatedTarget {
    RatedTarget _rt;
    __m128i _sse;
  };
} // anonymous namespace

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
  const CEQuiz<taNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  const taNumber* const PTR_RESTRICT pPriors = quiz.GetPriorMants();
  {
    const char *PTR_RESTRICT pCacheLine = SRCast::CPtr<char>(pPriors + _iFirst);
    _mm_prefetch(pCacheLine, _MM_HINT_NTA);
    _mm_prefetch(pCacheLine + SRCpuInfo::_cacheLineBytes, _MM_HINT_NTA);
  }
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<taNumber>&>(task.GetBaseEngine());
  const GapTracker<TPqaId> &PTR_RESTRICT gt = engine.GetTargetGaps();
  RatedTarget *PTR_RESTRICT pRatings = task.ModRatings() + _iWorker;
  RatedTarget *PTR_RESTRICT pTempRatings = task.ModTempRatings() + _iWorker;
  _pCounters = task.ModCounters() + _iWorker * cnBuckets;
  TPqaId *const PTR_RESTRICT pOffsets = task.ModOffsets() + _iWorker * cnBuckets;

  // Radix sort based on: http://stereopsis.com/radix.html , http://www.codercorner.com/RadixSortRevisited.htm

  ZeroCounters();
  TPqaId iSelLim = _iFirst;

  constexpr uint32_t nBytesAhead = (SRCpuInfo::_cacheLineBytes << 1);

  //NOTE: vectorization of 4 amounts at once would require conflict detection for the counters to increment, therefore
  //  I do not expect it to give a performance benifit.

  // pass 0 with flipping
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    //TODO: unroll, then prefetch once in several priors depending on sizeof(taNumber)
    _mm_prefetch(SRCast::CPtr<char>(pPriors + i) + nBytesAhead, _MM_HINT_NTA);
    if (gt.IsGap(i)) {
      continue;
    }
    const TPqaAmount prob = pPriors[i].ToAmount();
    if (prob <= 0) {
      continue;
    }
    const TPqaAmount flipped = Flip(prob);
    SseRatedTarget srt;
    srt._rt._iTarget = i;
    srt._rt._prob = flipped;
    _mm_stream_si128(SRCast::Ptr<__m128i>(pRatings + iSelLim), srt._sse);
    iSelLim++;
    _pCounters[SRCast::U64FromF64(flipped) & 0xff]++;
  }

  // passes 1 to 7 without flipping or unflipping
  for (uint8_t i = 1; i<=7; i++) {
    pOffsets[0] = _iFirst;
    for (size_t j = 1; j < cnBuckets; j++) {
      pOffsets[j] = pOffsets[j - 1] + _pCounters[j - 1];
    }
    ZeroCounters();
    for (TPqaId j = _iFirst; j < iSelLim; j++) {
      SseRatedTarget srt;
      srt._sse = _mm_stream_load_si128(SRCast::CPtr<__m128i>(pRatings + j));
      const uint64_t iProb = SRCast::U64FromF64(srt._rt._prob);
      _mm_stream_si128(SRCast::Ptr<__m128i>(pTempRatings + pOffsets[(iProb >> ((i - 1) << 3)) & 0xff]), srt._sse);
      _pCounters[(iProb >> (i << 3)) & 0xff]++;
    }
    std::swap(pRatings, pTempRatings);
  }

  //NOTE: pRatings and pTempRatings are swapped by pass 7 above, so current pRatings correspond to original pTempRatings
  // pass 8 with unflipping
  pOffsets[0] = _iFirst;
  for (size_t j = 1; j < cnBuckets; j++) {
    pOffsets[j] = pOffsets[j - 1] + _pCounters[j - 1];
  }
  for (TPqaId j = _iFirst; j < iSelLim; j++) {
    SseRatedTarget srt;
    srt._sse = _mm_stream_load_si128(SRCast::CPtr<__m128i>(pRatings + j));
    const uint64_t iProb = SRCast::U64FromF64(srt._rt._prob);
    srt._rt._prob = Unflip(srt._rt._prob);
    _mm_stream_si128(SRCast::Ptr<__m128i>(pTempRatings + pOffsets[iProb >> 56]), srt._sse);
  }

  // Put sentinel item
  SseRatedTarget sentinel;
  sentinel._rt._iTarget = cInvalidPqaId;
  sentinel._rt._prob = 0;
  _mm_stream_si128(SRCast::Ptr<__m128i>(pTempRatings + iSelLim), sentinel._sse);
  _mm_sfence();
}

} // namespace ProbQA
