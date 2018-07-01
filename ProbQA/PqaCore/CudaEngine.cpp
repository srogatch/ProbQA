// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CudaEngine.h"
#include "../PqaCore/PqaException.h"
#include "../PqaCore/CudaStreamPool.h"
#include "../PqaCore/CudaQuiz.h"
#include "../PqaCore/ErrorHelper.h"
#include "../PqaCore/CudaMemPool.h"
#include "../PqaCore/CudaPersistence.h"

using namespace SRPlat;

namespace ProbQA {

#define CUELOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, _pLogger.load(std::memory_order_acquire))

template<typename taNumber> CudaEngine<taNumber>::CudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi)
  : BaseCudaEngine(engDef, pKbFi),
  _sA(size_t(engDef._dims._nQuestions) * engDef._dims._nAnswers * engDef._dims._nTargets),
  _mD(size_t(engDef._dims._nQuestions) * engDef._dims._nTargets),
  _vB(engDef._dims._nTargets)
{
  const TPqaId nQuestions = _dims._nQuestions;
  const TPqaId nAnswers = _dims._nAnswers;
  const TPqaId nTargets = _dims._nTargets;

  const size_t nSAItems = int64_t(nQuestions) * nAnswers * nTargets;
  const size_t nMDItems = int64_t(nQuestions) * nTargets;
  const size_t nVBItems = nTargets;

  //TODO: this locks CUDA device for the duration of file operations
  CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
  CudaStream cuStr = _cspNb.Acquire();
  if (pKbFi == nullptr) { // init
    InitStatisticsKernel<taNumber> isk;
    isk._init1 = taNumber(engDef._initAmount);
    isk._initSqr = taNumber(isk._init1 * isk._init1);
    isk._initMD = taNumber(isk._initSqr * nAnswers);
    isk._nSAItems = nSAItems;
    isk._nMDItems = nMDItems;
    isk._nVBItems = nVBItems;
    isk._psA = _sA.Get();
    isk._pmD = _mD.Get();
    isk._pvB = _vB.Get();
    isk.Run(GetKlc(), cuStr.Get());
    CUDA_MUST(cudaGetLastError());
    CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
  } else { // load
    CudaPersistence cuPer{_cFileBufSize, pKbFi, cuStr.Get()};
    cuPer.ReadFile(_sA.Get(), sizeof(taNumber) * nSAItems);
    cuPer.ReadFile(_mD.Get(), sizeof(taNumber) * nMDItems);
    cuPer.ReadFile(_vB.Get(), sizeof(taNumber) * nVBItems);
  }
  AfterStatisticsInit(pKbFi);
  BceUpdateWithDimensions(cuStr.Get());
  CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
}

template<typename taNumber> PqaError CudaEngine<taNumber>::TrainSpec(const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs, const TPqaId iTarget, const TPqaAmount amount)
{
  (void)nQuestions;
  (void)pAQs;
  (void)iTarget;
  (void)amount;
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> TPqaId CudaEngine<taNumber>::ResumeQuizSpec(PqaError& err, const TPqaId nAnswered,
  const AnsweredQuestion* const pAQs)
{
  (void)nAnswered;
  (void)pAQs;
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CudaEngine<taNumber>::AddQsTsSpec(const TPqaId nQuestions,
  AddQuestionParam *pAqps, const TPqaId nTargets, AddTargetParam *pAtps)
{
  (void)nQuestions;
  (void)pAqps;
  (void)nTargets;
  (void)pAtps;
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> PqaError CudaEngine<taNumber>::CompactSpec(CompactionResult &cr) {
  (void)cr;
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> void CudaEngine<taNumber>::UpdateWithDimensions() {
  CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
  CudaStream cuStr = _cspNb.Acquire();
  BceUpdateWithDimensions(cuStr.Get());
  //TODO: implement - adjust CUDA memory pools, if any
  CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
}

template<typename taNumber> TPqaId CudaEngine<taNumber>::StartQuiz(PqaError& err) {
  try {
    constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
    if (!_maintSwitch.TryEnterSpecific<msMode>()) {
      err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform"
        " regular-only mode operation (Start quiz) because current mode is not regular"
        " (but maintenance/shutdown?)."));
      return cInvalidPqaId;
    }
    MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

    SRObjectMPP<CudaQuiz<taNumber>> spQuiz(_memPool, this);
    const TPqaId quizId = AssignQuiz(spQuiz.Get());
    try {
      StartQuizKernel<taNumber> sqk;
      sqk._nTargets = _dims._nTargets;
      sqk._pPriorMants = spQuiz.Get()->GetPriorMants();
      sqk._pQAsked = spQuiz.Get()->GetQAsked();
      sqk._pvB = _vB.Get();
      sqk._pTargetGaps = DevTargetGaps();
      {
        CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
        CudaStream cuStr = _cspNb.Acquire();
        SRRWLock<false> rwl(_rws);
        sqk.Run(GetKlc(), cuStr.Get());
        CUDA_MUST(cudaGetLastError());
        CudaMain::FlushWddm(cuStr.Get());
        CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
      }
    }
    CATCH_TO_ERR_SET(err);
    if (!err.IsOk()) {
      UnassignQuiz(quizId);
      return cInvalidPqaId;
    }
    spQuiz.Detach();
    return quizId;
  }
  CATCH_TO_ERR_SET(err);
  return cInvalidPqaId;
}

template<typename taNumber> TPqaId CudaEngine<taNumber>::NextQuestionSpec(PqaError& err, BaseQuiz *pBaseQuiz) {
  CudaQuiz<taNumber> *pQuiz = static_cast<CudaQuiz<taNumber>*>(pBaseQuiz);
  try {
    const TPqaId nQuestions = _dims._nQuestions;
    const TPqaId nAnswers = _dims._nAnswers;
    const TPqaId nTargets = _dims._nTargets;

    NextQuestionKernel<taNumber> nqk;
    nqk._nThreadsPerBlock = GetKlc().FixBlockSize(nTargets>>2);
    //nqk._nThreadsPerBlock = GetKlc().FixBlockSize(
    ///  uint64_t(GetKlc()._cdp.multiProcessorCount) * GetKlc()._cdp.maxThreadsPerMultiProcessor / nQuestions);
    //nqk._nThreadsPerBlock = GetKlc().FixBlockSize(nTargets);
    nqk._nBlocks = uint32_t(std::min(uint64_t(GetKlc()._cdp.multiProcessorCount)
      * GetKlc()._cdp.maxThreadsPerMultiProcessor / nqk._nThreadsPerBlock,
      uint64_t(nQuestions)));
    CudaMPArray<uint8_t> storage(GetCuMp(), SRSimd::_cNBytes // padding
      + SRSimd::GetPaddedBytes(sizeof(taNumber) * nQuestions) // totals
      + SRSimd::GetPaddedBytes(sizeof(taNumber) * nTargets * nqk._nBlocks) // posteriors
      + SRSimd::GetPaddedBytes(sizeof(taNumber) * nTargets * nqk._nBlocks) // _pInvD
      + SRSimd::GetPaddedBytes(sizeof(CudaAnswerMetrics<taNumber>) * nAnswers * nqk._nBlocks) // _pAnsMets
    );
    nqk._nQuestions = nQuestions;
    nqk._nAnswers = nAnswers;
    nqk._nTargets = nTargets;
    nqk._psA = _sA.Get();
    nqk._pmD = _mD.Get();
    nqk._pQAsked = pQuiz->GetQAsked();
    nqk._pPriorMants = pQuiz->GetPriorMants();
    nqk._pQuestionGaps = DevQuestionGaps();
    nqk._pTargetGaps = DevTargetGaps();
    nqk._pTotals = reinterpret_cast<taNumber*>(SRSimd::AlignPtr(storage.Get()));
    nqk._pPosteriors = reinterpret_cast<taNumber*>(SRSimd::AlignPtr(nqk._pTotals + nQuestions));
    nqk._pInvD = reinterpret_cast<taNumber*>(SRSimd::AlignPtr(nqk._pPosteriors + nTargets * nqk._nBlocks));
    nqk._pAnsMets = reinterpret_cast<CudaAnswerMetrics<taNumber>*>(SRSimd::AlignPtr(
      nqk._pInvD + nTargets * nqk._nBlocks));

    SRSmartMPP<taNumber> totals(_memPool, nQuestions);
    {
      CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
      CudaStream cuStr = _cspNb.Acquire();
      {
        SRRWLock<false> rwl(_rws);
        nqk.Run(cuStr.Get());
        CUDA_MUST(cudaGetLastError());
        CudaMain::FlushWddm(cuStr.Get());
        CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
      }
      CUDA_MUST(cudaMemcpyAsync(totals.Get(), nqk._pTotals, sizeof(taNumber)*nQuestions, cudaMemcpyDeviceToHost,
        cuStr.Get()));
      CudaMain::FlushWddm(cuStr.Get());
      CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
    }
    //// Analyze the totals so to select the next question
    struct QuestionInfo {
      TPqaId _iQuestion;
      taNumber _priority;
      bool operator<(const QuestionInfo& qi) const {
        return _priority < qi._priority;
      }
    };
    SRSmartMPP<QuestionInfo> heap(_memPool, nQuestions);
    //TODO: replace with Accumulator<taNumber>
    double grandTotal = 0;
    TPqaId nInHeap = 0;
    for (TPqaId i = 0; i < nQuestions; i++) {
      if (_questionGaps.IsGap(i) /*|| SRBitHelper::Test(pQuiz->GetQAsked(), i)*/) {
        continue;
      }
      taNumber priority = totals.Get()[i];
      if (priority < 0 || !std::isfinite(priority)) {
        CUELOG(Error) << "Got priority " << priority;
        continue;
      }
      heap.Get()[nInHeap]._iQuestion = i;
      heap.Get()[nInHeap]._priority = priority;
      grandTotal += priority;
      nInHeap++;
    }
    if (nInHeap == 0) {
      err = PqaError(PqaErrorCode::QuestionsExhausted, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Found no unasked"
        " question that is not in a gap."));
      return cInvalidPqaId;
    }
    std::make_heap(heap.Get(), heap.Get() + nInHeap);
    const double selected = (grandTotal * SRFastRandom::ThreadLocal().Generate<uint64_t>())
      / std::numeric_limits<uint64_t>::max();
    double poppedSum = heap.Get()[0]._priority;
    while (poppedSum < selected && nInHeap > 1) {
      std::pop_heap(heap.Get(), heap.Get() + nInHeap);
      nInHeap--;
      poppedSum += heap.Get()[0]._priority;
    }
    const TPqaId iQuestion = heap.Get()[0]._iQuestion;
    pQuiz->SetActiveQuestion(iQuestion);
    _nQuestionsAsked.fetch_add(1, std::memory_order_relaxed);
    return iQuestion;
  } CATCH_TO_ERR_SET(err);
  return cInvalidPqaId;
}

template<typename taNumber> TPqaId CudaEngine<taNumber>::ListTopTargetsSpec(PqaError& err, BaseQuiz *pBaseQuiz,
  const TPqaId maxCount, RatedTarget *pDest)
{
  try {
    CudaQuiz<taNumber> *pQuiz = static_cast<CudaQuiz<taNumber>*>(pBaseQuiz);
    const TPqaId nTargets = _dims._nTargets;
    SRSmartMPP<RatedTarget> all(_memPool, nTargets);
    SRSmartMPP<taNumber> priors(_memPool, nTargets);
    {
      CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
      CudaStream cuStr = _cspNb.Acquire();
      CUDA_MUST(cudaMemcpyAsync(priors.Get(), pQuiz->GetPriorMants(), sizeof(taNumber)*nTargets,
        cudaMemcpyDeviceToHost, cuStr.Get()));
      CudaMain::FlushWddm(cuStr.Get());
      CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
    }
    TPqaId nInHeap = 0;
    for (TPqaId i = 0; i < nTargets; i++) {
      if (_targetGaps.IsGap(i)) {
        continue;
      }
      all.Get()[nInHeap]._iTarget = i;
      all.Get()[nInHeap]._prob = priors.Get()[i];
      nInHeap++;
    }
    std::make_heap(all.Get(), all.Get() + nInHeap);
    const TPqaId nListed = std::min(maxCount, nInHeap);
    for (TPqaId i = 0; i < nListed; i++) {
      pDest[i] = all.Get()[0];
      std::pop_heap(all.Get(), all.Get() + nInHeap);
      nInHeap--;
    }
    err.Release();
    return nListed;
  }
  CATCH_TO_ERR_SET(err);
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CudaEngine<taNumber>::RecordQuizTargetSpec(BaseQuiz *pBaseQuiz,
  const TPqaId iTarget, const TPqaAmount amount)
{
  try {
    CudaQuiz<taNumber> *pQuiz = static_cast<CudaQuiz<taNumber>*>(pBaseQuiz);
    RecordQuizTargetKernel<taNumber> rqtk;
    rqtk._nAQs = pQuiz->GetAnswers().size();
    const AnsweredQuestion *pAQs = pQuiz->GetAnswers().data();
    CudaMPArray<CudaAnsweredQuestion> devAQs(GetCuMp(), std::max(rqtk._nAQs, _dims._nQuestions));
    SRSmartMPP<CudaAnsweredQuestion> hostAQs(_memPool, std::max(rqtk._nAQs, _dims._nQuestions));
    for (TPqaId i = 0; i < rqtk._nAQs; i++) {
      hostAQs.Get()[i]._iQuestion = pAQs[i]._iQuestion;
      hostAQs.Get()[i]._iAnswer = pAQs[i]._iAnswer;
    }
    std::sort(hostAQs.Get(), hostAQs.Get() + rqtk._nAQs);

    rqtk._iTarget = iTarget;
    rqtk._amount = taNumber(amount);
    rqtk._twoB = 2 * rqtk._amount;
    rqtk._bSquare = rqtk._amount * rqtk._amount;
    rqtk._nAnswers = _dims._nAnswers;
    rqtk._nTargets = _dims._nTargets;
    rqtk._nQuestions = _dims._nQuestions;
    rqtk._pAQs = devAQs.Get();
    rqtk._pmD = _mD.Get();
    rqtk._psA = _sA.Get();
    rqtk._pvB = _vB.Get();
    {
      CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
      CudaStream cuStr = _cspNb.Acquire();
      CUDA_MUST(cudaMemcpyAsync(devAQs.Get(), hostAQs.Get(), sizeof(CudaAnsweredQuestion) * rqtk._nAQs,
        cudaMemcpyHostToDevice, cuStr.Get()));
      CudaMain::FlushWddm(cuStr.Get());
      CUDA_MUST(cudaStreamSynchronize(cuStr.Get())); // minimize the time we are holding the read-write lock
      SRRWLock<true> rwl(_rws);
      rqtk.Run(GetKlc(), cuStr.Get());
      CUDA_MUST(cudaGetLastError());
      CudaMain::FlushWddm(cuStr.Get());
      CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
    }
    return PqaError();
  }
  CATCH_TO_ERR_RETURN;
}

template<typename taNumber> PqaError CudaEngine<taNumber>::DestroyQuiz(BaseQuiz *pQuiz) {
  // Report error if the object is not of type CEQuiz<taNumber>
  CudaQuiz<taNumber> *pSpecQuiz = dynamic_cast<CudaQuiz<taNumber>*>(pQuiz);
  if (pSpecQuiz == nullptr) {
    if (pQuiz == nullptr) {
      return PqaError();
    }
    return PqaError(PqaErrorCode::WrongRuntimeType, new WrongRuntimeTypeErrorParams(typeid(*pQuiz).name()),
      SRString::MakeUnowned(SR_FILE_LINE "Wrong runtime type of a quiz detected in an attempt to destroy it."));
  }
  SRCheckingRelease(_memPool, pSpecQuiz);
  return PqaError();
}

template<typename taNumber> PqaError CudaEngine<taNumber>::DestroyStatistics() {
  _sA.EarlyRelease();
  _mD.EarlyRelease();
  _vB.EarlyRelease();
  return PqaError();
}

template<typename taNumber> PqaError CudaEngine<taNumber>::SaveStatistics(KBFileInfo &kbfi) {
  const TPqaId nQuestions = _dims._nQuestions;
  const TPqaId nAnswers = _dims._nAnswers;
  const TPqaId nTargets = _dims._nTargets;

  const size_t nSAItems = int64_t(nQuestions) * nAnswers * nTargets;
  const size_t nMDItems = int64_t(nQuestions) * nTargets;
  const size_t nVBItems = nTargets;

  CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
  CudaStream cuStr = _cspNb.Acquire();
  CudaPersistence cuPer{_cFileBufSize, &kbfi, cuStr.Get()};
  try {
    cuPer.WriteFile(_sA.Get(), sizeof(taNumber) * nSAItems);
    cuPer.WriteFile(_mD.Get(), sizeof(taNumber) * nMDItems);
    cuPer.WriteFile(_vB.Get(), sizeof(taNumber) * nVBItems);
    return PqaError();
  }
  CATCH_TO_ERR_RETURN;
}

//// Instantiations
template class CudaEngine<float>;

} // namespace ProbQA
