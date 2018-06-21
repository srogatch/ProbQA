// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CudaEngine.h"
#include "../PqaCore/PqaException.h"
#include "../PqaCore/CudaStreamPool.h"
#include "../PqaCore/CudaQuiz.h"
#include "../PqaCore/ErrorHelper.h"

using namespace SRPlat;

namespace ProbQA {

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
  } else { // load
    if (std::fread(_sA.Get(), sizeof(taNumber), nSAItems, pKbFi->_sf.Get()) != nSAItems) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(
        SR_FILE_LINE " Can't read cube A from file.")).ThrowMoving();
    }
    _sA.Prefetch(cuStr.Get(), 0, nSAItems, _iDevice);
    if (std::fread(_mD.Get(), sizeof(taNumber), nMDItems, pKbFi->_sf.Get()) != nMDItems) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(
        SR_FILE_LINE " Can't read matrix D from file.")).ThrowMoving();
    }
    _mD.Prefetch(cuStr.Get(), 0, nMDItems, _iDevice);
    if (std::fread(_vB.Get(), sizeof(taNumber), nVBItems, pKbFi->_sf.Get()) != nVBItems) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(
        SR_FILE_LINE " Can't read vector B from file.")).ThrowMoving();
    }
    _vB.Prefetch(cuStr.Get(), 0, nVBItems, _iDevice);
  }
  AfterStatisticsInit(pKbFi);
  CopyGapsToDevice(cuStr.Get());
  CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
}

template<typename taNumber> PqaError CudaEngine<taNumber>::TrainSpec(const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs, const TPqaId iTarget, const TPqaAmount amount)
{
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> TPqaId CudaEngine<taNumber>::ResumeQuizSpec(PqaError& err, const TPqaId nAnswered,
  const AnsweredQuestion* const pAQs)
{
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
  return cInvalidPqaId;
}

template<typename taNumber> TPqaId CudaEngine<taNumber>::ListTopTargetsSpec(PqaError& err, BaseQuiz *pBaseQuiz,
  const TPqaId maxCount, RatedTarget *pDest)
{
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CudaEngine<taNumber>::RecordQuizTargetSpec(BaseQuiz *pBaseQuiz,
  const TPqaId iTarget, const TPqaAmount amount)
{
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> PqaError CudaEngine<taNumber>::AddQsTsSpec(const TPqaId nQuestions,
  AddQuestionParam *pAqps, const TPqaId nTargets, AddTargetParam *pAtps)
{
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> PqaError CudaEngine<taNumber>::CompactSpec(CompactionResult &cr) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> PqaError CudaEngine<taNumber>::SaveStatistics(KBFileInfo &kbfi) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> PqaError CudaEngine<taNumber>::DestroyQuiz(BaseQuiz *pQuiz) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> PqaError CudaEngine<taNumber>::DestroyStatistics() {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA engine is being implemented.")));
}

template<typename taNumber> void CudaEngine<taNumber>::UpdateWithDimensions() {
  CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
  CudaStream cuStr = _cspNb.Acquire();
  CopyGapsToDevice(cuStr.Get());
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
      CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
      CudaStream cuStr = _cspNb.Acquire();
      StartQuizKernel<taNumber> sqk;
      sqk._nTargets = _dims._nTargets;
      sqk._pPriorMants = spQuiz.Get()->GetPriorMants();
      sqk._pQAsked = reinterpret_cast<uint32_t*>(spQuiz.Get()->GetQAsked());
      sqk._pvB = _vB.Get();

      {
        SRRWLock<false> rwl(_rws);
        sqk.Run(GetKlc(), cuStr.Get());
        CUDA_MUST(cudaGetLastError());
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
    nqk._nThreadsPerBlock = GetKlc().FixBlockSize(nTargets);
    nqk._nBlocks = uint32_t(std::min(uint64_t(GetKlc()._cdp.multiProcessorCount)
      * GetKlc()._cdp.maxThreadsPerMultiProcessor / nqk._nThreadsPerBlock,
      uint64_t(nQuestions)));
    CudaArray<uint8_t, true> storage(SRSimd::_cNBytes // padding
      + SRSimd::GetPaddedBytes(sizeof(taNumber) * nQuestions) // totals
      + SRSimd::GetPaddedBytes(sizeof(taNumber) * nTargets * nqk._nBlocks) // posteriors
      + SRSimd::GetPaddedBytes(sizeof(taNumber) * nTargets * nqk._nBlocks) // _pInvD
      + SRSimd::GetPaddedBytes(sizeof(CudaAnswerMetrics<taNumber>) * nqk._nBlocks * nAnswers) // _pAnsMets
    );
    nqk._nQuestions = nQuestions;
    nqk._nAnswers = nAnswers;
    nqk._nTargets = nTargets;
    nqk._psA = _sA.Get();
    nqk._pmD = _mD.Get();
    nqk._pQAsked = reinterpret_cast<uint32_t*>(pQuiz->GetQAsked());
    nqk._pPriorMants = pQuiz->GetPriorMants();
    nqk._pQuestionGaps = DevQuestionGaps();
    nqk._pTargetGaps = DevTargetGaps();
    nqk._pTotals = reinterpret_cast<taNumber*>(SRSimd::AlignPtr(storage.Get()));
    nqk._pPosteriors = reinterpret_cast<taNumber*>(SRSimd::AlignPtr(nqk._pTotals + nQuestions));
    nqk._pInvD = reinterpret_cast<taNumber*>(SRSimd::AlignPtr(nqk._pPosteriors + nTargets * nqk._nBlocks));
    nqk._pAnsMets = reinterpret_cast<CudaAnswerMetrics<taNumber>*>(SRSimd::AlignPtr(
      nqk._pInvD + nTargets * nqk._nBlocks));

    CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
    CudaStream cuStr = _cspNb.Acquire();
    {
      SRRWLock<false> rwl(_rws);
      nqk.Run(cuStr.Get());
      CUDA_MUST(cudaGetLastError());
      CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
    }
  } CATCH_TO_ERR_SET(err);
  return cInvalidPqaId;
}

//// Instantiations
template class CudaEngine<float>;

} // namespace ProbQA
