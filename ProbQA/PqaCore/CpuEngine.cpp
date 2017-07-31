// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/DoubleNumber.h"
#include "../PqaCore/PqaException.h"
#include "../PqaCore/CETrainTask.h"
#include "../PqaCore/ErrorHelper.h"
#include "../PqaCore/CETask.h"
#include "../PqaCore/CETrainSubtaskDistrib.h"
#include "../PqaCore/CETrainSubtaskAdd.h"
#include "../PqaCore/CETrainTaskNumSpec.h"
#include "../PqaCore/CEQuiz.h"
#include "../PqaCore/CECreateQuizOperation.h"

using namespace SRPlat;

namespace ProbQA {

#define CELOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, _pLogger.load(std::memory_order_acquire))

template<typename taNumber> CpuEngine<taNumber>::CpuEngine(const EngineDefinition& engDef) : BaseCpuEngine(engDef) {
  if (_dims._nAnswers < cMinAnswers || _dims._nQuestions < cMinQuestions || _dims._nTargets < cMinTargets)
  {
    throw PqaException(PqaErrorCode::InsufficientEngineDimensions, new InsufficientEngineDimensionsErrorParams(
      _dims._nAnswers, cMinAnswers, _dims._nQuestions, cMinQuestions, _dims._nTargets, cMinTargets));
  }

  taNumber initAmount(engDef._initAmount);
  //// Init cube A: A[ao][q][t] is weight for answer option |ao| for question |q| for target |t|
  _sA.resize(size_t(_dims._nAnswers));
  for (size_t i = 0, iEn= size_t(_dims._nAnswers); i < iEn; i++) {
    _sA[i].resize(size_t(_dims._nQuestions));
    for (size_t j = 0, jEn= size_t(_dims._nQuestions); j < jEn; j++) {
      _sA[i][j].Resize<false>(size_t(_dims._nTargets));
      _sA[i][j].FillAll<false>(initAmount);
    }
  }

  //// Init matrix D: D[q][t] is the sum of weigths over all answers for question |q| for target |t|. In the other
  ////   words, D[q][t] is A[0][q][t] + A[1][q][t] + ... + A[K-1][q][t], where K is the number of answer options.
  //// Note that D is subject to summation errors, thus its regular recomputation is desired.
  taNumber initMD = initAmount * _dims._nAnswers;
  _mD.resize(size_t(_dims._nQuestions));
  for (size_t i = 0, iEn=size_t(_dims._nQuestions); i < iEn; i++) {
    _mD[i].Resize<false>(size_t(_dims._nTargets));
    _mD[i].FillAll<false>(initMD);
  }

  //// Init vector B: the sums of weights over all trainings for each target
  _vB.Resize<false>(size_t(_dims._nTargets));
  _vB.FillAll<false>(initAmount);

  _questionGaps.GrowTo(_dims._nQuestions);
  _targetGaps.GrowTo(_dims._nTargets);

  //throw PqaException(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
  //  "CpuEngine<taNumber>::CpuEngine(const EngineDefinition& engDef)")));
}

template<typename taNumber> CpuEngine<taNumber>::~CpuEngine() {
  PqaError pqaErr = Shutdown();
  if (!pqaErr.isOk() && pqaErr.GetCode() != PqaErrorCode::ObjectShutDown) {
    CELOG(Error) << "Failed CpuEngine::Shutdown(): " << pqaErr.ToString(true);
  }
}

template<typename taNumber> PqaError CpuEngine<taNumber>::SetLogger(ISRLogger *pLogger) {
  if (pLogger == nullptr) {
    pLogger = SRDefaultLogger::Get();
  }
  _pLogger.store(pLogger, std::memory_order_release);
  return PqaError();
}

template<typename taNumber> PqaError CpuEngine<taNumber>::Shutdown(const char* const saveFilePath) {
  if (!_maintSwitch.Shutdown()) {
    // Return an error saying that the engine seems already shut down.
    SRMessageBuilder mbMsg("MaintenanceSwitch seems already shut down.");
    if (saveFilePath != nullptr) {
      mbMsg(" Not saving file: ")(saveFilePath);
    }
    return PqaError(PqaErrorCode::ObjectShutDown, new ObjectShutDownErrorParams(SRString::MakeUnowned(
      "CpuEngine<taNumber>::Shutdown()")), mbMsg.GetOwnedSRString());
  }
  // By this moment, all operations must have shut down and no new operations can be started.

  if (saveFilePath != nullptr) {
    //TODO: implement
  }

  //TODO: check the order - perhaps some releases should happen while the workers are still operational

  //// Shutdown worker threads
  _tpWorkers.RequestShutdown();

  //// Release quizzes
  for (size_t i = 0; i < _quizzes.size(); i++) {
    delete _quizzes[i];
  }
  _quizzes.clear();
  _quizGaps.Compact(0);

  //// Release KB
  _sA.clear();
  _mD.clear();
  _vB.Clear();
  _questionGaps.Compact(0);
  _targetGaps.Compact(0);
  _dims._nAnswers = _dims._nQuestions = _dims._nTargets = 0;

  //// Release memory pool
  _memPool.FreeAllChunks();

  return PqaError();
}

template<> void CpuEngine<DoubleNumber>::InitTrainTaskNumSpec(CETrainTaskNumSpec<DoubleNumber>& numSpec,
  const TPqaAmount amount) 
{
  const double dAmount = to_double(amount);
  numSpec._collAddend = numSpec._fullAddend = _mm256_set1_pd(dAmount);
  // Colliding addend: amount is added twice to _mD[iQuestion][iTarget] .
  numSpec._collAddend.m256d_f64[1] += dAmount;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::TrainInternal(const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs, const TPqaId iTarget, const TPqaAmount amount)
{
  if (nQuestions < 0) {
    return PqaError(PqaErrorCode::NegativeCount, new NegativeCountErrorParams(nQuestions), SRString::MakeUnowned(
      "|nQuestions| must be non-negative."));
  }
  if (amount <= 0) {
    return PqaError(PqaErrorCode::NonPositiveAmount, new NonPositiveAmountErrorParams(amount), SRString::MakeUnowned(
      "|amount| must be positive."));
  }

  PqaError resErr;
  const SRThreadPool::TThreadCount nWorkers = _tpWorkers.GetWorkerCount();
  //// Do a single allocation for all needs. Allocate memory out of locks.
  // For proper alignment, the data must be laid out in the decreasing order of item alignments.
  const size_t ttLastOffs = std::max(sizeof(CETrainSubtaskDistrib<taNumber>), sizeof(CETrainSubtaskAdd<taNumber>))
    * nWorkers;
  const size_t ttPrevOffs = ttLastOffs + sizeof(std::atomic<TPqaId>) * nWorkers;
  const size_t nTotalBytes = ttPrevOffs + sizeof(TPqaId) * SRCast::ToSizeT(nQuestions);
  SRSmartMPP<TMemPool, uint8_t> commonBuf(_memPool, nTotalBytes);

  CETrainTask<taNumber> trainTask(this, nWorkers, iTarget, pAQs);
  trainTask._prev = reinterpret_cast<TPqaId*>(commonBuf.Get() + ttPrevOffs);
  trainTask._last = reinterpret_cast<std::atomic<TPqaId>*>(commonBuf.Get() + ttLastOffs);
  InitTrainTaskNumSpec(trainTask._numSpec, amount);
  //TODO: vectorize/parallelize
  for (size_t i = 0; i < nWorkers; i++) {
    new(trainTask._last + i) std::atomic<TPqaId>(cInvalidPqaId);
  }
  // &trainTask, &nWorkers
  auto&& ttLastFinally = SRMakeFinally([&pLast = trainTask._last, &nWorkers]{
    //TODO: vectorize/parallelize
    for (size_t i = 0; i < nWorkers; i++) {
      pLast[i].~atomic();
    }
  }); (void)ttLastFinally; // prevent warning C4189

  { // Scope for the locks
    MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
    // Can't move dimensions-related code out of this lock because this operation can be run in maintenance mode too.
    SRRWLock<true> rwl(_rws);

    //// This code must be reader-writer locked, because we are validating the input before modifying the KB, so noone
    ////   must change or read the KB in between.

    if (iTarget < 0 || iTarget >= _dims._nTargets) {
      const TPqaId nKB = _dims._nTargets;
      rwl.EarlyRelease();
      return PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iTarget, 0, nKB - 1),
        SRString::MakeUnowned("Target index is not in KB range."));
    }

    if (_targetGaps.IsGap(iTarget)) {
      rwl.EarlyRelease();
      return PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iTarget), SRString::MakeUnowned(
        "Target index is not in KB (but rather at a gap)."));
    }

    //// Distribute the AQs into buckets with the number of buckets divisable by the number of workers.
    resErr = SplitAndRunSubtasks<CETrainSubtaskDistrib<taNumber>>(trainTask, nQuestions, commonBuf.Get(),
      [&](CETrainSubtaskDistrib<taNumber> *pCurSt, const size_t curStart, const size_t nextStart)
    {
      new (pCurSt) CETrainSubtaskDistrib<taNumber>(&trainTask, pAQs + curStart, pAQs + nextStart);
    });
    if (!resErr.isOk()) {
      return resErr;
    }

    // Update the KB with the given training data.
    resErr = RunWorkerOnlySubtasks<CETrainSubtaskAdd<taNumber>>(trainTask, commonBuf.Get());
    if (!resErr.isOk()) {
      return resErr;
    }

    _vB[iTarget] += amount;

    // This method should increase the counter of questions asked by the number of questions in this training.
    _nQuestionsAsked += nQuestions;
  }

  return PqaError();
}

template<typename taNumber> PqaError CpuEngine<taNumber>::Train(const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs, const TPqaId iTarget, const TPqaAmount amount)
{
  try {
    return TrainInternal(nQuestions, pAQs, iTarget, amount);
  }
  CATCH_TO_ERR_RETURN;
}

template<typename taNumber> template<typename taOperation> TPqaId CpuEngine<taNumber>::CreateQuizInternal(
  taOperation &op)
{
  try {
    struct LTask : public CEBaseTask {
      CEQuiz<taNumber> *_pQuiz;
    public: // methods
      explicit LTask(CpuEngine<taNumber> *pCe) : CEBaseTask(pCe) { }
    } task(this);

    const auto msMode = MaintenanceSwitch::Mode::Regular;
    if (!_maintSwitch.TryEnterSpecific<msMode>()) {
      op._err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned("Can't perform regular-only mode"
        " operation because current mode is not regular (but maintenance/shutdown?)."));
      return cInvalidPqaId;
    }
    MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);
    // So long as this constructor only needs the number of questions and targets, it can be out of _rws because
    //   _maintSwitch guards engine dimensions.
    op._pQuiz = new CEQuiz<taNumber>(this);
    TPqaId quizId;
    {
      SRLock<SRCriticalSection> csl(_csQuizReg);
      quizId = _quizGaps.Acquire();
      if (quizId >= TPqaId(_quizzes.size())) {
        assert(quizId == TPqaId(_quizzes.size()));
        _quizzes.emplace_back(nullptr);
      }
      _quizzes[SRCast::ToSizeT(quizId)] = op._pQuiz;
    }
    //TODO: perhaps get rid of taOperation::_pQuiz , and keep quiz pointer only in the task.
    task._pQuiz = op._pQuiz;
    {
      SRRWLock<false> rwl(_rws);

      //// Copy prior likelihoods for targets
      auto&& lstCopyMants = SRMakeLambdaSubtask(&task, [](const SRBaseSubtask &subtask) {
        auto& task = static_cast<const LTask&>(*subtask.GetTask());
        auto& engine = static_cast<const CpuEngine<taNumber>&>(*task.GetEngine());
        SRUtils::Copy256<false, false>(task._pQuiz->GetTlhMants(), engine._vB.Get(),
          SRSimd::VectsFromBytes(engine._dims._nTargets * sizeof(taNumber)));
      });
      _tpWorkers.Enqueue(&lstCopyMants);

      auto&& lstZeroExps = SRMakeLambdaSubtask(&task, [](const SRBaseSubtask &subtask) {
        auto& task = static_cast<const LTask&>(*subtask.GetTask());
        auto& engine = static_cast<const CpuEngine<taNumber>&>(*task.GetEngine());
        SRUtils::FillZeroVects<false>(reinterpret_cast<__m256i*>(task._pQuiz->GetTlhExps()),
          SRSimd::VectsFromBytes(engine._dims._nTargets * sizeof(CEQuiz<taNumber>::TExponent)));
      });
      _tpWorkers.Enqueue(&lstZeroExps);

      auto&& lstZeroQAsked = SRMakeLambdaSubtask(&task, [](const SRBaseSubtask &subtask) {
        auto& task = static_cast<const LTask&>(*subtask.GetTask());
        auto& engine = static_cast<const CpuEngine<taNumber>&>(*task.GetEngine());
        SRUtils::FillZeroVects<true>(task._pQuiz->GetQAsked(), SRSimd::VectsFromBits(engine._dims._nQuestions));
      });
      _tpWorkers.Enqueue(&lstZeroQAsked);

      task.WaitComplete();

      op.MaybeUpdatePriorsWithAnsweredQuestions(this);
    }
    if (!op._err.isOk()) {
      delete op._pQuiz;
      SRLock<SRCriticalSection> csl(_csQuizReg);
      _quizzes[SRCast::ToSizeT(quizId)] = nullptr;
      _quizGaps.Release(quizId);
      return cInvalidPqaId;
    }
    return quizId;
  }
  CATCH_TO_ERR_SET(op._err);
  return cInvalidPqaId;
}

template<typename taNumber> void CpuEngine<taNumber>::UpdatePriorsWithAnsweredQuestions(
  CECreateQuizResume<taNumber>& resumeOp)
{
  //// Sequential code (single-threaded) for reference
  //NOTE: it may be better to iterate by targets first instead, so to apply all multiplications for the first
  //  target and then move on to the next target. This involves 1 unsequential memory access per answered question
  //  application, while if we iterate first by questions, each question application involves 2 memory accesses: load
  //  and store.
  //const TPqaId nTargets = _dims._nTargets;
  //taNumber *pTargProb = resumeOp._pQuiz->GetTargProbs();
  //TPqaId i = 0;
  //for (; i + 1 < resumeOp._nQuestions; i++) {
  //  const AnsweredQuestion& aq = resumeOp._pAQs[i];
  //  for (TPqaId j = 0; j < nTargets; j++) {
  //    // Multiplier compensation is less robust than summation of logarithms, but it's substantially faster and is
  //    //   supported by AVX2. The idea is to make the multipliers equal to 1 in the average case p[j]=1/M, where M is
  //    //   the number of targets.
  // //FIXME: this will blow to infinity the top most likely targets, making them all equal, which is highly undesirable
  //    pTargProb[j] *= (nTargets * _sA[aq._iAnswer][aq._iQuestion][j] / _mD[aq._iQuestion][j]);
  //  }
  //}
  //taNumber sum(0); //TODO: instead, sort then sum
  //const AnsweredQuestion& aq = resumeOp._pAQs[i];
  //for (TPqaId j = 0; j < nTargets; j++) {
  //  taNumber product = pTargProb[j] * (nTargets * _sA[aq._iAnswer][aq._iQuestion][j] / _mD[aq._iQuestion][j]);
  //  pTargProb[j] = product;
  //  sum += product; //TODO: assign to a bucket instead
  //}
  //for (TPqaId j = 0; j < nTargets; j++) {
  //  pTargProb[j] /= sum;
  //}

  //TODO: implement
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::StartQuiz(PqaError& err) {
  CECreateQuizStart<taNumber> startOp(err);
  return CreateQuizInternal(startOp);
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::ResumeQuiz(PqaError& err, const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs) 
{
  CECreateQuizResume<taNumber> resumeOp(err, nQuestions, pAQs);
  return CreateQuizInternal(resumeOp);
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::NextQuestion(PqaError& err, const TPqaId iQuiz) {
  (void)iQuiz; //TODO: remove when implemented
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::NextQuestion")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer) {
  (void)iQuiz; (void)iAnswer; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RecordAnswer")));
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::ListTopTargets(PqaError& err, const TPqaId iQuiz,
  const TPqaId maxCount, RatedTarget *pDest) 
{
  (void)iQuiz; (void)maxCount; (void)pDest; //TODO: remove when implemented
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::ListTopTargets")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RecordQuizTarget(const TPqaId iQuiz, const TPqaId iTarget,
  const TPqaAmount amount) 
{
  (void)iQuiz; (void)iTarget; (void)amount; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RecordQuizTarget")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::ReleaseQuiz(const TPqaId iQuiz) {
  (void)iQuiz; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::ReleaseQuiz")));
}


template<typename taNumber> PqaError CpuEngine<taNumber>::SaveKB(const char* const filePath, const bool bDoubleBuffer) {
  (void)filePath; (void)bDoubleBuffer; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::SaveKB")));
}

template<typename taNumber> uint64_t CpuEngine<taNumber>::GetTotalQuestionsAsked(PqaError& err) {
  err.Release();
  SRRWLock<false> rwsl(_rws);
  return _nQuestionsAsked;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::StartMaintenance(const bool forceQuizes) {
  (void)forceQuizes; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::StartMaintenance")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::FinishMaintenance() {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::FinishMaintenance")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::AddQuestions(TPqaId nQuestions, AddQuestionParam *pAqps) {
  (void)nQuestions; (void)pAqps; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddQuestions")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::AddTargets(TPqaId nTargets, AddTargetParam *pAtps) {
  (void)nTargets; (void)pAtps; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddTargets")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds)
{
  (void)nQuestions; (void)pQIds; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RemoveQuestions")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RemoveTargets(const TPqaId nTargets, const TPqaId *pTIds) {
  (void)nTargets; (void)pTIds; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RemoveTargets")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::Compact(CompactionResult &cr) {
  (void)cr; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::Compact")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::ReleaseCompactionResult(CompactionResult &cr) {
  (void)cr; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::ReleaseCompactionResult")));
}

//// Instantiations
template class CpuEngine<DoubleNumber>;

} // namespace ProbQA
