// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEListTopTargetsAlgorithm.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/CEQuiz.h"
#include "../PqaCore/RatingsHeap.h"
#include "../PqaCore/CEHeapifyPriorsTask.h"
#include "../PqaCore/CEHeapifyPriorsSubtaskMake.h"
#include "../PqaCore/CERadixSortRatingsTask.h"
#include "../PqaCore/CERadixSortRatingsSubtaskSort.h"

using namespace SRPlat;

namespace ProbQA {

//TODO: because this class is not likely to have specialized methods, to avoid excessive listing of all the supported
//  template arguments here, move the implementation to fwd/decl/h header-only idiom.
template class CEListTopTargetsAlgorithm<SRDoubleNumber>;

template<typename taNumber> CEListTopTargetsAlgorithm<taNumber>::CEListTopTargetsAlgorithm(PqaError &PTR_RESTRICT err, 
  CpuEngine<taNumber> &PTR_RESTRICT engine, const CEQuiz<taNumber> &PTR_RESTRICT quiz, const TPqaId maxCount,
  RatedTarget *PTR_RESTRICT pDest) : _err(err), _pEngine(&engine), _pQuiz(&quiz), _maxCount(maxCount), _pDest(pDest),
  _nWorkers(engine.GetWorkers().GetWorkerCount()), _nTargets(engine.GetDims()._nTargets)
{ }

template<typename taNumber> TPqaId CEListTopTargetsAlgorithm<taNumber>::RunHeapifyBased() {
  // This algorithm is optimized for small number of top targets to list. A separate branch is needed if substantial
  //   part of all targets is to be listed. That branch could use radix sort in separate threads, then heap merge.
  SRMemTotal mtCommon;
  const SRMemItem miSubtasks(_nWorkers * SRMaxSizeof<CEHeapifyPriorsSubtaskMake<taNumber>>::value, SRMemPadding::None,
    mtCommon);
  const SRMemItem miSplit(SRPoolRunner::CalcSplitMemReq(_nWorkers), SRMemPadding::None, mtCommon);
  const SRMemItem miPieceLimits(_nWorkers * sizeof(TPqaId), SRMemPadding::None, mtCommon);
  const SRMemItem miHeadHeap(_nWorkers * sizeof(RatingsHeapItem), SRMemPadding::None, mtCommon);
  const SRMemItem miRatings(_nTargets * sizeof(RatedTarget), SRMemPadding::Both, mtCommon);

  SRSmartMPP<uint8_t> commonBuf(_pEngine->GetMemPool(), mtCommon._nBytes);
  SRPoolRunner pr(_pEngine->GetWorkers(), miSubtasks.BytePtr(commonBuf));

  SRPoolRunner::Split targSplit = SRPoolRunner::CalcSplit(miSplit.BytePtr(commonBuf), _nTargets, _nWorkers);
  TPqaId *const PTR_RESTRICT pPieceLimits = miPieceLimits.ToPtr<TPqaId>(commonBuf);
  RatedTarget *const PTR_RESTRICT pRatings = miRatings.ToPtr<RatedTarget>(commonBuf);
  {
    CEHeapifyPriorsTask<taNumber> hpTask(*_pEngine, *_pQuiz, pRatings, pPieceLimits);
    pr.RunPreSplit<CEHeapifyPriorsSubtaskMake<taNumber>>(hpTask, targSplit);
  }

  //NOTE: after this, the split is no more valid for launching subtasks
  targSplit.RecalcToStarts();
  const size_t *const PTR_RESTRICT pStarts = targSplit._pBounds;
  RatingsHeapItem *const PTR_RESTRICT pHeadHeap = miHeadHeap.ToPtr<RatingsHeapItem>(commonBuf);
  SRThreadCount nHhItems = 0;
  for (SRThreadCount i = 0; i < targSplit._nSubtasks; i++) {
    const TPqaId curFirst = pStarts[i];
    if (pPieceLimits[i] == curFirst) {
      continue;
    }
    pHeadHeap[nHhItems]._iSource = i;
    pHeadHeap[nHhItems]._prob = pRatings[curFirst]._prob;
    nHhItems++;
  }
  std::make_heap(pHeadHeap, pHeadHeap + nHhItems);

  for (TPqaId i = 0; i < _maxCount; i++) {
    assert(nHhItems >= 0);
    if (nHhItems == 0) {
      return i; // the number of targets actually listed
    }
    _pDest[i]._prob = pHeadHeap[0]._prob;
    const SRThreadCount curPiece = static_cast<SRThreadCount>(pHeadHeap[0]._iSource);
    const TPqaId pieceStart = pStarts[curPiece];
    _pDest[i]._iTarget = pRatings[pieceStart]._iTarget;

    const TPqaId pieceLim = pPieceLimits[curPiece];
    assert(pieceStart + 1 <= pieceLim);
    if (pieceStart + 1 == pieceLim) {
      // This piece has been exhausted
      std::pop_heap(pHeadHeap, pHeadHeap + nHhItems);
      nHhItems--;
      // Can be here for consistency, but currently never used further.
      // pPieceLimits[curPiece]--;
      continue;
    }

    std::pop_heap(pRatings + pieceStart, pRatings + pieceLim);
    pPieceLimits[curPiece]--; // pieceLim is no longer valid after this point

    pHeadHeap[0]._prob = pRatings[pieceStart]._prob;
    SRHeapHelper::Down(pHeadHeap, pHeadHeap + nHhItems);
  }

  return _maxCount;
}

template<typename taNumber> TPqaId CEListTopTargetsAlgorithm<taNumber>::RunRadixSortBased() {
  // This algorithm is good when the requested number of targets is large enough w.r.t. the total number of targets.
  SRMemTotal mtCommon;
  const SRMemItem miSubtasks(_nWorkers * SRMaxSizeof<CERadixSortRatingsSubtaskSort<taNumber>>::value,
    SRMemPadding::None, mtCommon);
  const SRMemItem miSplit(SRPoolRunner::CalcSplitMemReq(_nWorkers), SRMemPadding::None, mtCommon);
  const SRMemItem miHeadHeap(_nWorkers * sizeof(RatingsHeapItem), SRMemPadding::None, mtCommon);

  // Each piece must have a room for a sentinel item denoting end of the piece.
  const size_t ratingsSize = (_nTargets + _nWorkers) * sizeof(RatedTarget);
  const SRMemItem miRatings(ratingsSize, SRMemPadding::Both, mtCommon);
  const SRMemItem miTempRatings(ratingsSize, SRMemPadding::Both, mtCommon);

  static_assert(_cnRadixSortBuckets * sizeof(TPqaId) % SRSimd::_cNBytes == 0 , "Assumed no need for padding.");
  const size_t bucketsBytes = _nWorkers * _cnRadixSortBuckets * sizeof(TPqaId);
  const SRMemItem miCounters(bucketsBytes, SRMemPadding::Both, mtCommon);
  const SRMemItem miOffsets(bucketsBytes, SRMemPadding::Both, mtCommon);

  SRSmartMPP<uint8_t> commonBuf(_pEngine->GetMemPool(), mtCommon._nBytes);
  SRPoolRunner pr(_pEngine->GetWorkers(), miSubtasks.BytePtr(commonBuf));

  SRPoolRunner::Split targSplit = SRPoolRunner::CalcSplit(miSplit.BytePtr(commonBuf), _nTargets, _nWorkers);
  const RatedTarget *PTR_RESTRICT pcRatings;
  {
    RatedTarget *const PTR_RESTRICT pRatings = miRatings.ToPtr<RatedTarget>(commonBuf);
    pcRatings = pRatings;

    CERadixSortRatingsTask<taNumber> rsrTask(*_pEngine, *_pQuiz, pRatings, miTempRatings.ToPtr<RatedTarget>(commonBuf),
      miCounters.ToPtr<TPqaId>(commonBuf), miOffsets.ToPtr<TPqaId>(commonBuf));
    pr.RunPreSplit<CERadixSortRatingsSubtaskSort<taNumber>>(rsrTask, targSplit);
  }

  //NOTE: after this, the split is no more valid for launching subtasks
  targSplit.RecalcToStarts();
  const size_t *const PTR_RESTRICT pStarts = targSplit._pBounds;
  RatingsHeapItem *const PTR_RESTRICT pHeadHeap = miHeadHeap.ToPtr<RatingsHeapItem>(commonBuf);
  SRThreadCount nHhItems = 0;
  for (SRThreadCount i = 0; i < targSplit._nSubtasks; i++) {
    const TPqaId curFirst = pStarts[i] + i;
    const TPqaAmount curProb = pcRatings[curFirst]._prob;
    if (curProb <= 0) { // sentinel item
      continue;
    }
    pHeadHeap[nHhItems]._iSource = curFirst;
    pHeadHeap[nHhItems]._prob = curProb;
    nHhItems++;
  }
  std::make_heap(pHeadHeap, pHeadHeap + nHhItems);
  
  for (TPqaId i = 0; i < _maxCount; i++) {
    assert(nHhItems >= 0);
    if (nHhItems == 0) {
      return i; // the number of targets actually listed
    }
    _pDest[i]._prob = pHeadHeap[0]._prob;
    _pDest[i]._iTarget = pcRatings[pHeadHeap[0]._iSource]._iTarget;
    
    const TPqaId nextSource = ++pHeadHeap[0]._iSource;
    const TPqaAmount nextProb = pcRatings[nextSource]._prob;

    if (nextProb <= 0) {
      // This piece has been exhausted
      std::pop_heap(pHeadHeap, pHeadHeap + nHhItems);
      nHhItems--;
      continue;
    }

    pHeadHeap[0]._prob = nextProb;
    SRHeapHelper::Down(pHeadHeap, pHeadHeap + nHhItems);
  }

  return _maxCount;
}

} // namespace ProbQA
