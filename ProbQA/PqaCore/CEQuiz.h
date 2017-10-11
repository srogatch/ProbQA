// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEQuiz.decl.h"
#include "../PqaCore/CEDivTargPriorsSubtask.h"
#include "../PqaCore/CERecordAnswerTask.h"
#include "../PqaCore/CERecordAnswerSubtaskMul.h"
#include "../PqaCore/CpuEngine.h"

namespace ProbQA {

//////////////////////////////// CEBaseQuiz implementation /////////////////////////////////////////////////////////////

inline CEBaseQuiz::CEBaseQuiz(BaseCpuEngine *pEngine) : _pEngine(pEngine) {
  using namespace SRPlat;
  const EngineDimensions& dims = _pEngine->GetDims();
  const size_t nQuestions = SRPlat::SRCast::ToSizeT(dims._nQuestions);
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);

  SRMemTotal mtCommon;
  SRMemItem<__m256i> miIsQAsked(SRPlat::SRSimd::VectsFromBits(nQuestions), SRPlat::SRMemPadding::Both, mtCommon);
  SRMemItem<TExponent> miExponents(nTargets, SRPlat::SRMemPadding::Both, mtCommon);
  // First allocate all the memory so to revert if anything fails.
  SRSmartMPP<uint8_t> commonBuf(_pEngine->GetMemPool(), mtCommon._nBytes);
  // Must be the first memory block, because it's used for releasing the memory
  _isQAsked = miIsQAsked.Ptr(commonBuf);
  _pTlhExps = miExponents.Ptr(commonBuf);
  // As all the memory is allocated, safely proceed with finishing construction of CEBaseQuiz object.
  commonBuf.Detach();
}

inline CEBaseQuiz::~CEBaseQuiz() {
  using namespace SRPlat;
  //NOTE: engine dimensions must not change during lifetime of the quiz because below we must provide the same number
  //  of targets and questions.
  const EngineDimensions& dims = _pEngine->GetDims();
  const size_t nQuestions = SRPlat::SRCast::ToSizeT(dims._nQuestions);
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);

  SRMemTotal mtCommon;
  SRMemItem<__m256i> miIsQAsked(SRPlat::SRSimd::VectsFromBits(nQuestions), SRPlat::SRMemPadding::Both, mtCommon);
  SRMemItem<TExponent> miExponents(nTargets, SRPlat::SRMemPadding::Both, mtCommon);
  _pEngine->GetMemPool().ReleaseMem(_isQAsked, mtCommon._nBytes);
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
  SRBitHelper::Set(GetQAsked(), _activeQuestion);
  _activeQuestion = cInvalidPqaId;

  // Update prior probabilities in the quiz
  CpuEngine<taNumber> &PTR_RESTRICT engine = *GetEngine();
  const EngineDimensions &PTR_RESTRICT dims = engine.GetDims();
  const SRThreadCount nWorkers = engine.GetWorkers().GetWorkerCount();

  SRMemTotal mtCommon;
  const SRByteMem miSubtasks(nWorkers *std::max(SRBucketSummatorPar<taNumber>::_cSubtaskMemReq,
    SRMaxSizeof<CERecordAnswerSubtaskMul<SRDoubleNumber>, CEDivTargPriorsSubtask<CERecordAnswerTask<taNumber>>>::value
  ), SRMemPadding::None, mtCommon);
  const SRByteMem miSplit(SRPoolRunner::CalcSplitMemReq(nWorkers), SRMemPadding::Both, mtCommon);
  //TODO: refactor to Kahan summation
  const SRByteMem miBuckets(SRBucketSummatorPar<taNumber>::GetMemoryRequirementBytes(nWorkers),
    SRMemPadding::Both, mtCommon);

  SRSmartMPP<uint8_t> commonBuf(engine.GetMemPool(), mtCommon._nBytes);
  SRPoolRunner pr(engine.GetWorkers(), miSubtasks.BytePtr(commonBuf));
  SRBucketSummatorPar<taNumber> bsp(nWorkers, miBuckets.BytePtr(commonBuf));

  const TPqaId nTargetVects = SRSimd::VectsFromComps<taNumber>(dims._nTargets);
  const SRPoolRunner::Split targSplit = SRPoolRunner::CalcSplit(miSplit.BytePtr(commonBuf), nTargetVects, nWorkers);

  CERecordAnswerTask<taNumber> raTask(engine, *this, _answers.back(), bsp);
  {
    SRRWLock<false> rwl(engine.GetRws());
    pr.RunPreSplit<CERecordAnswerSubtaskMul<SRDoubleNumber>>(raTask, targSplit);
  }
  raTask._sumPriors.Set1(bsp.ComputeSum(pr));
  // Divide the likelihoods by their sum calculated above
  pr.RunPreSplit<CEDivTargPriorsSubtask<CERecordAnswerTask<taNumber>>>(raTask, targSplit);
  return PqaError();
}

} // namespace ProbQA
