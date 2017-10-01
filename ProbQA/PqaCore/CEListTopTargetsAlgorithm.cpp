// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEListTopTargetsAlgorithm.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/CEQuiz.h"
#include "../PqaCore/RatingsHeap.h"
#include "../PqaCore/CEHeapifyPriorsTask.h"
#include "../PqaCore/PqaRange.h"

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
  const SRMemItem miSubtasks(_nWorkers * SRMaxSizeof</*TODO: make_heap subtask here*/>::value,
    SRMemPadding::None, mtCommon);
  const SRMemItem miRatings(_nTargets * sizeof(RatedTarget), SRMemPadding::Both, mtCommon);
  const SRMemItem miHeadHeap(_nWorkers * sizeof(RatingsHeapItem), SRMemPadding::None, mtCommon);
  const SRMemItem miSourceInfos(_nWorkers * sizeof(PqaRange), SRMemPadding::None, mtCommon);
  //TODO: radix sort temporary array of ratings, and buckets here

  SRSmartMPP<uint8_t> commonBuf(_pEngine->GetMemPool(), mtCommon._nBytes);
  SRPoolRunner pr(_pEngine->GetWorkers(), miSubtasks.BytePtr(commonBuf));

  CEHeapifyPriorsTask<taNumber> hpTask(*_pEngine, *_pQuiz, miRatings.ToPtr<RatedTarget>(commonBuf),
    miSourceInfos.ToPtr<PqaRange>(commonBuf));
  //TODO: implement, assuming normalized priors

  _err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    SR_FILE_LINE "Heapify-based top targets selection")));
  return cInvalidPqaId;
}

template<typename taNumber> TPqaId CEListTopTargetsAlgorithm<taNumber>::RunRadixSortBased() {
  _err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    SR_FILE_LINE "RadixSort-based top targets selection")));
  return cInvalidPqaId;
}

} // namespace ProbQA
