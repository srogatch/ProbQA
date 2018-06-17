// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CudaEngine.h"
#include "../PqaCore/PqaException.h"
#include "../PqaCore/CudaStreamPool.h"

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
    CudaDeviceLock cdl = CudaMain::SetDevice(_iDevice);
    isk.Run(GetKlc(), cuStr.Get());
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
  CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
}

template<typename taNumber> PqaError CudaEngine<taNumber>::TrainSpec(const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs, const TPqaId iTarget, const TPqaAmount amount)
{

}

template<typename taNumber> TPqaId CudaEngine<taNumber>::ResumeQuizSpec(PqaError& err, const TPqaId nAnswered,
  const AnsweredQuestion* const pAQs)
{

}

template<typename taNumber> TPqaId CudaEngine<taNumber>::NextQuestionSpec(PqaError& err, BaseQuiz *pBaseQuiz) {
}

template<typename taNumber> TPqaId CudaEngine<taNumber>::ListTopTargetsSpec(PqaError& err, BaseQuiz *pBaseQuiz,
  const TPqaId maxCount, RatedTarget *pDest)
{

}

template<typename taNumber> PqaError CudaEngine<taNumber>::RecordQuizTargetSpec(BaseQuiz *pBaseQuiz,
  const TPqaId iTarget, const TPqaAmount amount)
{
}

template<typename taNumber> PqaError CudaEngine<taNumber>::AddQsTsSpec(const TPqaId nQuestions,
  AddQuestionParam *pAqps, const TPqaId nTargets, AddTargetParam *pAtps)
{
}

template<typename taNumber> PqaError CudaEngine<taNumber>::CompactSpec(CompactionResult &cr) {

}

template<typename taNumber> PqaError CudaEngine<taNumber>::SaveStatistics(KBFileInfo &kbfi) {

}

template<typename taNumber> PqaError CudaEngine<taNumber>::DestroyQuiz(BaseQuiz *pQuiz) {
}

template<typename taNumber> PqaError CudaEngine<taNumber>::DestroyStatistics() {
}

template<typename taNumber> TPqaId CudaEngine<taNumber>::StartQuiz(PqaError& err) {
}

//// Instantiations
template class CudaEngine<float>;

} // namespace ProbQA
