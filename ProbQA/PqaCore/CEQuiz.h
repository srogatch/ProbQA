// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEQuiz.decl.h"
#include "../PqaCore/CERecordAnswerTask.h"
#include "../PqaCore/CERecordAnswerSubtaskMul.h"
#include "../PqaCore/CpuEngine.h"

namespace ProbQA {

//////////////////////////////// CEBaseQuiz implementation /////////////////////////////////////////////////////////////

inline CEBaseQuiz::CEBaseQuiz(BaseCpuEngine *pEngine) : _pEngine(pEngine) {
  const EngineDimensions& dims = _pEngine->GetDims();
  const size_t nQuestions = SRPlat::SRCast::ToSizeT(dims._nQuestions);
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);
  auto& memPool = _pEngine->GetMemPool();

  // First allocate all the memory so to revert if anything fails.
  SRPlat::SRSmartMPP<__m256i> smppIsQAsked(memPool, SRPlat::SRSimd::VectsFromBits(nQuestions));
  SRPlat::SRSmartMPP<TExponent> smppExponents(memPool, nTargets);

  // As all the memory is allocated, safely proceed with finishing construction of CEBaseQuiz object.
  _pTlhExps = smppExponents.Detach();
  _isQAsked = smppIsQAsked.Detach();
}

inline CEBaseQuiz::~CEBaseQuiz() {
  const EngineDimensions& dims = GetBaseEngine()->GetDims();
  const size_t nQuestions = SRPlat::SRCast::ToSizeT(dims._nQuestions);
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);
  auto& memPool = GetBaseEngine()->GetMemPool();
  //NOTE: engine dimensions must not change during lifetime of the quiz because below we must provide the same number
  //  of targets and questions.
  memPool.ReleaseMem(_pTlhExps, sizeof(*_pTlhExps) * nTargets);
  memPool.ReleaseMem(_isQAsked, SRPlat::SRBitHelper::GetAlignedSizeBytes(nQuestions));
}

//////////////////////////////// CEQuiz implementation /////////////////////////////////////////////////////////////////

template<typename taNumber> inline CpuEngine<taNumber>* CEQuiz<taNumber>::GetEngine() const {
  return static_cast<CpuEngine<taNumber>*>(GetBaseEngine());
}

template<typename taNumber> CEQuiz<taNumber>::CEQuiz(CpuEngine<taNumber> *pEngine) : CEBaseQuiz(pEngine) {
  const EngineDimensions& dims = pEngine->GetDims();
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);
  auto& memPool = pEngine->GetMemPool();

  // First allocate all the memory so to revert if anything fails.
  SRPlat::SRSmartMPP<taNumber> smppMantissas(memPool, nTargets);

  // As all the memory is allocated, safely proceed with finishing construction of CEQuiz object.
  _pPriorMants = smppMantissas.Detach();
}

template<typename taNumber> CEQuiz<taNumber>::~CEQuiz() {
  const EngineDimensions& dims = GetBaseEngine()->GetDims();
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);
  auto& memPool = GetBaseEngine()->GetMemPool();
  //NOTE: engine dimensions must not change during lifetime of the quiz because below we must provide the same number
  //  of targets.
  memPool.ReleaseMem(_pPriorMants, sizeof(*_pPriorMants) * nTargets);
}

template<typename taNumber> inline PqaError CEQuiz<taNumber>::RecordAnswer(const TPqaId iAnswer) {
  if (_activeQuestion == cInvalidPqaId) {
    return PqaError(PqaErrorCode::NoQuizActiveQuestion, new NoQuizActiveQuestionErrorParams(iAnswer),
      SRPlat::SRString::MakeUnowned(SR_FILE_LINE "An attempt to record an answer in a quiz that doesn't have an active"
        "question"));
  }
  _answers.emplace_back(_activeQuestion, iAnswer);
  _activeQuestion = cInvalidPqaId;

  // Update prior probabilities in the quiz
  CpuEngine<taNumber> &PTR_RESTRICT engine = *GetEngine();
  const EngineDimensions &PTR_RESTRICT dims = engine.GetDims();
  const SRThreadCount nWorkers = engine.GetWorkers().GetWorkerCount();

  constexpr size_t subtasksOffs = 0;
  const size_t splitOffs = subtasksOffs + nWorkers *std::max({ SRBucketSummatorPar<taNumber>::_cSubtaskMemReq,
    SRMaxSizeof<CERecordAnswerSubtaskMul<SRDoubleNumber>>::value });
  const size_t bucketsOffs = SRSimd::GetPaddedBytes(splitOffs + SRPoolRunner::CalcSplitMemReq(nWorkers));
  const size_t nWithBuckets = SRSimd::GetPaddedBytes(bucketsOffs +
    SRBucketSummatorPar<taNumber>::GetMemoryRequirementBytes(nWorkers));
  const size_t totalBytes = nWithBuckets;

  SRSmartMPP<uint8_t> commonBuf(engine.GetMemPool(), totalBytes);
  SRPoolRunner pr(engine.GetWorkers(), commonBuf.Get() + subtasksOffs);
  SRBucketSummatorPar<taNumber> bsp(nWorkers, commonBuf.Get() + bucketsOffs);

  const TPqaId nTargetVects = SRSimd::VectsFromComps<double>(dims._nTargets);
  const SRPoolRunner::Split targSplit = SRPoolRunner::CalcSplit(commonBuf.Get() + splitOffs, nTargetVects, nWorkers);

  CERecordAnswerTask<taNumber> raTask(engine, *this, _answers.back(), bsp);
  {
    SRRWLock<false> rwl(engine.GetRws());
    pr.SplitAndRunSubtasks<CERecordAnswerSubtaskMul<SRDoubleNumber>>(raTask, nTargetVects);
  }
  raTask._sumPriors.Set1(bsp.ComputeSum(pr));
  //TODO: divide the likelihoods by their sum calculated above

  return PqaError();
}

} // namespace ProbQA
