// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/PqaException.h"
#include "../PqaCore/CETrainTask.h"
#include "../PqaCore/ErrorHelper.h"
#include "../PqaCore/CETask.h"
#include "../PqaCore/CETrainSubtaskDistrib.h"
#include "../PqaCore/CETrainSubtaskAdd.h"
#include "../PqaCore/CETrainTaskNumSpec.h"
#include "../PqaCore/CEQuiz.h"
#include "../PqaCore/CECreateQuizOperation.h"
#include "../PqaCore/CENormPriorsTask.h"
#include "../PqaCore/CENormPriorsSubtaskMax.h"
#include "../PqaCore/CENormPriorsSubtaskCorrSum.h"
#include "../PqaCore/CENormPriorsSubtaskDiv.h"
#include "../PqaCore/CEEvalQsTask.h"
#include "../PqaCore/CEEvalQsSubtaskConsider.h"

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
  //// Init cube A: A[q][ao][t] is weight for answer option |ao| for question |q| for target |t|
  _sA.resize(SRCast::ToSizeT(_dims._nQuestions));
  for (size_t i = 0, iEn= SRCast::ToSizeT(_dims._nQuestions); i < iEn; i++) {
    _sA[i].resize(SRCast::ToSizeT(_dims._nAnswers));
    for (size_t k = 0, kEn= SRCast::ToSizeT(_dims._nAnswers); k < kEn; k++) {
      _sA[i][k].Resize<false>(SRCast::ToSizeT(_dims._nTargets));
      _sA[i][k].FillAll<false>(initAmount);
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
  if (!pqaErr.IsOk() && pqaErr.GetCode() != PqaErrorCode::ObjectShutDown) {
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

template<> void CpuEngine<SRDoubleNumber>::InitTrainTaskNumSpec(CETrainTaskNumSpec<SRDoubleNumber>& numSpec,
  const TPqaAmount amount) 
{
  const double dAmount = SRCast::ToDouble(amount);
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
  const SRThreadCount nWorkers = _tpWorkers.GetWorkerCount();
  //// Do a single allocation for all needs. Allocate memory out of locks.
  // For proper alignment, the data must be laid out in the decreasing order of item alignments.
  const size_t ttLastOffs = nWorkers * SRMaxSizeof<CETrainSubtaskDistrib<taNumber>, CETrainSubtaskAdd<taNumber>>::value;
  const size_t ttPrevOffs = ttLastOffs + sizeof(std::atomic<TPqaId>) * nWorkers;
  const size_t nTotalBytes = ttPrevOffs + sizeof(TPqaId) * SRCast::ToSizeT(nQuestions);
  SRSmartMPP<uint8_t> commonBuf(_memPool, nTotalBytes);

  CETrainTask<taNumber> trainTask(*this, nWorkers, iTarget, pAQs);
  //TODO: these are slow because threads share a cache line. It's not clear yet how to workaround this: the data is not
  //  per-source thread, but rather per target thread (after distribution).
  trainTask._prev = SRCast::Ptr<TPqaId>(commonBuf.Get() + ttPrevOffs);
  trainTask._last = SRCast::Ptr<std::atomic<TPqaId>>(commonBuf.Get() + ttLastOffs);
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
    
    //// The further code must be reader-writer locked, because we are validating the input before modifying the KB,
    ////   so noone must change or read the KB in between.
    SRRWLock<true> rwl(_rws);

    // Can't move dimensions-related code out of SRW lock because this operation can be run in maintenance mode too.
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

    SRPoolRunner pr(_tpWorkers, commonBuf.Get());

    //// Distribute the AQs into buckets with the number of buckets divisable by the number of workers.
    pr.SplitAndRunSubtasks<CETrainSubtaskDistrib<taNumber>>(trainTask, nQuestions, trainTask.GetWorkerCount(),
      [&](void *pStMem, SRThreadCount iWorker, int64_t iFirst, int64_t iLimit) {
        new (pStMem) CETrainSubtaskDistrib<taNumber>(&trainTask, pAQs + iFirst, pAQs + iLimit);
        (void)iWorker;
      }
    );
    resErr = trainTask.TakeAggregateError(SRString::MakeUnowned("Failed " SR_FILE_LINE));
    if (!resErr.IsOk()) {
      return resErr;
    }

    //// Update the KB with the given training data.
    pr.RunPerWorkerSubtasks<CETrainSubtaskAdd<taNumber>>(trainTask, trainTask.GetWorkerCount());
    resErr = trainTask.TakeAggregateError(SRString::MakeUnowned("Failed " SR_FILE_LINE));
    if (!resErr.IsOk()) {
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

template<typename taNumber> TPqaId CpuEngine<taNumber>::CreateQuizInternal(CECreateQuizOpBase &op) {
  try {
    struct NoSrwTask : public CETask {
      CEQuiz<taNumber> *_pQuiz;
    public: // methods
      explicit NoSrwTask(CpuEngine<taNumber> &ce) : CETask(ce, /*nWorkers*/ 3) { }
    } tNoSrw(*this); // Subtasks without SRW locked

    constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
    if (!_maintSwitch.TryEnterSpecific<msMode>()) {
      op._err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned("Can't perform regular-only mode"
        " operation (Start/Resume quiz) because current mode is not regular (but maintenance/shutdown?)."));
      return cInvalidPqaId;
    }
    MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

    // So long as this constructor only needs the number of questions and targets, it can be out of _rws because
    //   _maintSwitch guards engine dimensions in Regular mode (but not in Maintenance mode).
    tNoSrw._pQuiz = new CEQuiz<taNumber>(this);
    TPqaId quizId;
    {
      SRLock<SRCriticalSection> csl(_csQuizReg);
      quizId = _quizGaps.Acquire();
      if (quizId >= TPqaId(_quizzes.size())) {
        assert(quizId == TPqaId(_quizzes.size()));
        _quizzes.emplace_back(nullptr);
      }
      _quizzes[SRCast::ToSizeT(quizId)] = tNoSrw._pQuiz;
    }

    {
      auto&& lstZeroExps = SRMakeLambdaSubtask(&tNoSrw, [](const SRBaseSubtask &subtask) {
        auto& task = static_cast<const NoSrwTask&>(*subtask.GetTask());
        auto& engine = static_cast<const CpuEngine<taNumber>&>(task.GetBaseEngine());
        SRUtils::FillZeroVects<false>(SRCast::Ptr<__m256i>(task._pQuiz->GetTlhExps()),
          SRSimd::VectsFromBytes(engine._dims._nTargets * sizeof(CEQuiz<taNumber>::TExponent)));
      });
      auto&& lstSetQAsked = SRMakeLambdaSubtask(&tNoSrw, [&op](const SRBaseSubtask &subtask) {
        auto& task = static_cast<NoSrwTask&>(*subtask.GetTask());
        auto& engine = static_cast<const CpuEngine<taNumber>&>(task.GetBaseEngine());
        __m256i *pQAsked = task._pQuiz->GetQAsked();
        SRUtils::FillZeroVects<true>(pQAsked, SRSimd::VectsFromBits(engine._dims._nQuestions));
        if (op.IsResume()) {
          // Validate the indexes and set "is question asked" bits
          auto& resumeOp = static_cast<CECreateQuizResume<taNumber>&>(op);
          const EngineDimensions& dims = engine.GetDims();
          for (size_t i = 0; i<SRCast::ToSizeT(resumeOp._nAnswered); i++) {
            const TPqaId iQuestion = resumeOp._pAQs[i]._iQuestion;
            if (iQuestion < 0 || iQuestion >= dims._nQuestions) {
              task.AddError(PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iQuestion, 0,
                dims._nQuestions - 1), SRString::MakeUnowned(SR_FILE_LINE "Question index is not in KB range.")));
              return;
            }
            const TPqaId iAnswer = resumeOp._pAQs[i]._iAnswer;
            if (iAnswer < 0 || iAnswer >= dims._nAnswers) {
              task.AddError(PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iAnswer, 0,
                dims._nAnswers - 1), SRString::MakeUnowned(SR_FILE_LINE "Answer index is not in KB range.")));
              return;
            }
            *(SRCast::Ptr<uint8_t>(pQAsked) + (iQuestion >> 3)) |= (1ui8 << (iQuestion & 7));
          }
        }
      });
      auto&& lstAddAnswers = SRMakeLambdaSubtask(&tNoSrw, [&op](const SRBaseSubtask &subtask) {
        auto& resumeOp = static_cast<const CECreateQuizResume<taNumber>&>(op);
        auto& task = static_cast<const NoSrwTask&>(*subtask.GetTask());
        std::vector<AnsweredQuestion>& answers = task._pQuiz->ModAnswers();
        answers.insert(answers.end(), resumeOp._pAQs, resumeOp._pAQs + resumeOp._nAnswered);
      });

      SRTaskWaiter noSrwTaskWaiter(&tNoSrw);
      if(op.IsResume()) {
        _tpWorkers.Enqueue({&lstZeroExps, &lstSetQAsked, &lstAddAnswers}, tNoSrw);
      } else {
        _tpWorkers.Enqueue({&lstZeroExps, &lstSetQAsked}, tNoSrw);
      }

      {
        SRRWLock<false> rwl(_rws);
        // Copy prior likelihood mantissas for targets
        SRUtils::Copy256<false, false>(tNoSrw._pQuiz->GetTlhMants(), _vB.Get(),
          SRSimd::VectsFromBytes(_dims._nTargets * sizeof(taNumber)));
      }
    }
    op._err = tNoSrw.TakeAggregateError();
    if(op._err.IsOk()) {
      //// If it's "resume quiz" operation, update the prior likelihoods with the questions answered.
      op.UpdateLikelihoods(*this, *tNoSrw._pQuiz);
    }
    if (!op._err.IsOk()) {
      delete tNoSrw._pQuiz;
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

template<typename taNumber> TPqaId CpuEngine<taNumber>::StartQuiz(PqaError& err) {
  CECreateQuizStart startOp(err);
  return CreateQuizInternal(startOp);
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::ResumeQuiz(PqaError& err, const TPqaId nAnswered,
  const AnsweredQuestion* const pAQs) 
{
  if(nAnswered < 0) {
    err = PqaError(PqaErrorCode::NegativeCount, new NegativeCountErrorParams(nAnswered), SRString::MakeUnowned(
      "|nAnswered| must be non-negative."));
    return cInvalidPqaId;
  }
  CECreateQuizResume<taNumber> resumeOp(err, nAnswered, pAQs);
  return CreateQuizInternal(resumeOp);
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::NextQuestion(PqaError& err, const TPqaId iQuiz) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned("Can't perform regular-only mode"
      " operation (compute next question) because current mode is not regular (but maintenance/shutdown?)."));
    return cInvalidPqaId;
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  CEQuiz<taNumber> *pQuiz;
  {
    SRLock<SRCriticalSection> csl(_csQuizReg);
    const TPqaId nQuizzes = _quizzes.size();
    if(iQuiz < 0 || iQuiz >= nQuizzes) {
      csl.EarlyRelease();
      // For nQuizzes == 0, this may return [0;-1] range: we can't otherwise return an empty range because we return
      //   the range with both bounds inclusive.
      err = PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iQuiz, 0, nQuizzes - 1),
        SRString::MakeUnowned("Quiz index is not in quiz registry range."));
      return cInvalidPqaId;
    }
    if(_quizGaps.IsGap(iQuiz)) {
      csl.EarlyRelease();
      err = PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iQuiz), SRString::MakeUnowned(
        "Quiz index is not in the registry (but rather at a gap)."));
      return cInvalidPqaId;
    }
    pQuiz = _quizzes[iQuiz];
  }

  const SRThreadCount nWorkers = _tpWorkers.GetWorkerCount();
  const size_t subtasksOffs = 0;
  const size_t splitOffs = subtasksOffs + nWorkers * SRMaxSizeof< CENormPriorsSubtaskMax<taNumber>,
    CENormPriorsSubtaskCorrSum<taNumber>, CENormPriorsSubtaskDiv<taNumber>, CEEvalQsSubtaskConsider<taNumber> >::value;
  const size_t bucketsOffs = SRSimd::GetPaddedBytes(splitOffs + SRPoolRunner::CalcSplitMemReq(nWorkers));
  const size_t runLengthOffs = SRSimd::GetPaddedBytes(bucketsOffs + std::max(
    // Here we rely that GetMemoryRequirementBytes() returns SIMD-aligned number of bytes.
    SRBucketSummatorPar<taNumber>::GetMemoryRequirementBytes(nWorkers),
    //TODO: use sequential summator here instead.
    nWorkers * SRBucketSummatorPar<taNumber>::GetMemoryRequirementBytes(1) ));
  const size_t nWithRunLength = runLengthOffs + SRSimd::GetPaddedBytes(sizeof(taNumber) * _dims._nQuestions);
  SRSmartMPP<uint8_t> commonBuf(_memPool, nWithRunLength);

  SRPoolRunner pr(_tpWorkers, commonBuf.Get() + subtasksOffs);
  SRBucketSummatorPar<taNumber> bsp(nWorkers, commonBuf.Get() + bucketsOffs);

  { //TODO: move this block to a separate function, encapsulating local variables in a context structure?
    CENormPriorsTask<taNumber> normPriorsTask(*this, *pQuiz, bsp);
    const int64_t nTargetVects = SRMath::PosDivideRoundUp(_dims._nTargets, TPqaId(SRNumPack<taNumber>::_cnComps));
    const SRPoolRunner::Split targSplit = SRPoolRunner::CalcSplit(commonBuf.Get() + splitOffs, nTargetVects, nWorkers);

    { // The lifetime for maximum selection subtasks
      SRPoolRunner::Keeper<CENormPriorsSubtaskMax<taNumber>> kp = pr.RunPreSplit<CENormPriorsSubtaskMax<taNumber>>(
        normPriorsTask, targSplit);
      assert(kp.GetNSubtasks() == targSplit._nSubtasks);
      const SRThreadCount nResultVects = kp.GetNSubtasks() >> SRSimd::_cLogNComps64;
      const SRVectCompCount nTail = SRVectCompCount(kp.GetNSubtasks() - (nResultVects << SRSimd::_cLogNComps64));
      __m256i vMaxExps;
      auto fnFetch = [&](const SRVectCompCount at) { return kp.GetSubtask(at)->_maxExp; };
      if (nResultVects == 0) {
        SRSimd::ForTailI64(nTail, fnFetch, [&](const __m256i& vect) { vMaxExps = vect; },
          std::numeric_limits<int64_t>::min());
      }
      else {
        vMaxExps = _mm256_set_epi64x(kp.GetSubtask(3)->_maxExp, kp.GetSubtask(2)->_maxExp, kp.GetSubtask(1)->_maxExp,
          kp.GetSubtask(0)->_maxExp);
        for (SRThreadCount i = 1; i < nResultVects; i++) {
          const SRThreadCount iBase = (i << SRSimd::_cLogNComps64);
          const __m256i cand = _mm256_set_epi64x(kp.GetSubtask(iBase + 3)->_maxExp,
            kp.GetSubtask(iBase + 2)->_maxExp, kp.GetSubtask(iBase + 1)->_maxExp, kp.GetSubtask(iBase)->_maxExp);
          vMaxExps = SRSimd::MaxI64(vMaxExps, cand);
        }
        SRSimd::ForTailI64(nTail, fnFetch, [&](const __m256i& cand) { vMaxExps = SRSimd::MaxI64(vMaxExps, cand); },
          std::numeric_limits<int64_t>::min());
      }
      const int64_t fullMax = SRSimd::FullHorizMaxI64(vMaxExps);
      const int64_t highBound = taNumber::_cMaxExp + taNumber::_cExpOffs - SRMath::CeilLog2(_dims._nTargets) - 2;
      const int64_t minAllowed = std::numeric_limits<int64_t>::min() + highBound + 1;
      if (fullMax <= minAllowed) {
        err = PqaError(PqaErrorCode::I64Underflow, new I64UnderflowErrorParams(fullMax, minAllowed),
          SRString::MakeUnowned("Max exponent over the priors is too low. Are all the targets in gaps?"));
        return cInvalidPqaId;
      }
      normPriorsTask._corrExp = _mm256_set1_epi64x(highBound - fullMax); // so that fullMax + correction == highBound
    }

    // Correct the exponents towards the taNumber range, and calculate their sum
    pr.RunPreSplit<CENormPriorsSubtaskCorrSum<taNumber>>(normPriorsTask, targSplit);
    normPriorsTask._sumPriors.Set1(bsp.ComputeSum(pr));

    // Divide priors by their sum, so to get probabilities.
    pr.RunPreSplit<CENormPriorsSubtaskDiv<taNumber>>(normPriorsTask, targSplit);
  }

  {
    CEEvalQsTask<taNumber> evalQsTask(*this, *pQuiz, _dims._nTargets - _targetGaps.GetNGaps(),
      commonBuf.Get() + bucketsOffs, SRCast::Ptr<taNumber>(commonBuf.Get() + runLengthOffs));
    // Although there are no more subtasks which would use this split, it will be used for run-length analysis.
    const SRPoolRunner::Split questionSplit = SRPoolRunner::CalcSplit(commonBuf.Get() + splitOffs, _dims._nQuestions,
      nWorkers);
    SRRWLock<false> rwl(_rws);
    SRPoolRunner::Keeper<CEEvalQsSubtaskConsider<taNumber>> kp = pr.RunPreSplit<CEEvalQsSubtaskConsider<taNumber>>(
      evalQsTask, questionSplit);
  }

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
template class CpuEngine<SRDoubleNumber>;

} // namespace ProbQA
