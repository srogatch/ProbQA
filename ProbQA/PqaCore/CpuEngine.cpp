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
#include "../PqaCore/CEEvalQsTask.h"
#include "../PqaCore/CEEvalQsSubtaskConsider.h"
#include "../PqaCore/RatingsHeap.h"

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
    SRCheckingRelease(_memPool, _quizzes[i]);
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
    _nQuestionsAsked.fetch_add(nQuestions, std::memory_order_relaxed);
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
    SRObjectMPP<CEQuiz<taNumber>> spQuiz(_memPool, this);
    tNoSrw._pQuiz = spQuiz.Get();
    TPqaId quizId;
    {
      SRLock<SRCriticalSection> csl(_csQuizReg);
      quizId = _quizGaps.Acquire();
      if (quizId >= TPqaId(_quizzes.size())) {
        assert(quizId == TPqaId(_quizzes.size()));
        _quizzes.emplace_back(nullptr);
      }
      _quizzes[SRCast::ToSizeT(quizId)] = spQuiz.Get();
    }

    {
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
        _tpWorkers.Enqueue({&lstSetQAsked, &lstAddAnswers}, tNoSrw);
      } else {
        _tpWorkers.Enqueue(&lstSetQAsked);
      }
    }
    op._err = tNoSrw.TakeAggregateError();
    if(op._err.IsOk()) {
      // If it's "resume quiz" operation, update the prior likelihoods with the questions answered, and normalize the
      //   priors. If it's "start quiz" operation, just divide the priors by their sum.
      op.UpdateLikelihoods(*this, *spQuiz.Get());
    }
    if (!op._err.IsOk()) {
      spQuiz.EarlyRelease();
      SRLock<SRCriticalSection> csl(_csQuizReg);
      _quizzes[SRCast::ToSizeT(quizId)] = nullptr;
      _quizGaps.Release(quizId);
      return cInvalidPqaId;
    }
    spQuiz.Detach();
    return quizId;
  }
  CATCH_TO_ERR_SET(op._err);
  return cInvalidPqaId;
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::StartQuiz(PqaError& err) {
  CECreateQuizStart<taNumber> startOp(err);
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
  if (nAnswered == 0) {
    return StartQuiz(err);
  }
  CECreateQuizResume<taNumber> resumeOp(err, nAnswered, pAQs);
  return CreateQuizInternal(resumeOp);
}

template<typename taNumber> CEQuiz<taNumber>* CpuEngine<taNumber>::UseQuiz(PqaError& err, const TPqaId iQuiz) {
  SRLock<SRCriticalSection> csl(_csQuizReg);
  const TPqaId nQuizzes = _quizzes.size();
  if (iQuiz < 0 || iQuiz >= nQuizzes) {
    csl.EarlyRelease();
    // For nQuizzes == 0, this may return [0;-1] range: we can't otherwise return an empty range because we return
    //   the range with both bounds inclusive.
    err = PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iQuiz, 0, nQuizzes - 1),
      SRString::MakeUnowned(SR_FILE_LINE "Quiz index is not in quiz registry range."));
    return nullptr;
  }
  if (_quizGaps.IsGap(iQuiz)) {
    csl.EarlyRelease();
    err = PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iQuiz), SRString::MakeUnowned(
      SR_FILE_LINE "Quiz index is not in the registry (but rather at a gap)."));
    return nullptr;
  }
  return _quizzes[iQuiz];
}

template<typename taNumber> PqaError CpuEngine<taNumber>::NormalizePriors(CEQuiz<taNumber> &quiz, SRPoolRunner &pr,
  SRBucketSummatorPar<taNumber> &bsp, const SRPoolRunner::Split& targSplit)
{
  CENormPriorsTask<taNumber> normPriorsTask(*this, quiz, bsp);

  { // The lifetime for maximum selection subtasks
    SRPoolRunner::Keeper<CENormPriorsSubtaskMax<taNumber>> kp = pr.RunPreSplit<CENormPriorsSubtaskMax<taNumber>>(
      normPriorsTask, targSplit);
    assert(kp.GetNSubtasks() == targSplit._nSubtasks);
    const SRThreadCount nResultVects = kp.GetNSubtasks() >> SRSimd::_cLogNComps64;
    const SRVectCompCount nTail = kp.GetNSubtasks() & ((SRThreadCount(1) << SRSimd::_cLogNComps64) - 1);
    __m256i vMaxExps;
    auto fnFetch = [&kp, iBase = (nResultVects << SRSimd::_cLogNComps64)](const SRVectCompCount at) {
      return kp.GetSubtask(iBase + at)->_maxExp;
    };
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
      return PqaError(PqaErrorCode::I64Underflow, new I64UnderflowErrorParams(fullMax, minAllowed),
        SRString::MakeUnowned("Max exponent over the priors is too low. Are all the targets in gaps?"));
    }
    normPriorsTask._corrExp = _mm256_set1_epi64x(highBound - fullMax); // so that fullMax + correction == highBound
  }

  // Correct the exponents towards the taNumber range, and calculate their sum
  pr.RunPreSplit<CENormPriorsSubtaskCorrSum<taNumber>>(normPriorsTask, targSplit);
  normPriorsTask._sumPriors.Set1(bsp.ComputeSum(pr));

  // Divide priors by their sum, so to get probabilities.
  pr.RunPreSplit<CEDivTargPriorsSubtask<CENormPriorsTask<taNumber>>>(normPriorsTask, targSplit);

  return PqaError();
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::NextQuestion(PqaError& err, const TPqaId iQuiz) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned("Can't perform regular-only mode"
      " operation (compute next question) because current mode is not regular (but maintenance/shutdown?)."));
    return cInvalidPqaId;
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  CEQuiz<taNumber> *pQuiz = UseQuiz(err, iQuiz);
  if (pQuiz == nullptr) {
    assert(!err.IsOk());
    return cInvalidPqaId;
  }

  const SRThreadCount nWorkers = _tpWorkers.GetWorkerCount();
  constexpr size_t subtasksOffs = 0;
  const size_t splitOffs = subtasksOffs + nWorkers * SRMaxSizeof<CEEvalQsSubtaskConsider<taNumber> >::value;
  const size_t bucketsOffs = SRSimd::GetPaddedBytes(splitOffs + SRPoolRunner::CalcSplitMemReq(nWorkers));
  const size_t runLengthOffs = SRSimd::GetPaddedBytes(bucketsOffs +
    // Here we rely that SRBucketSummatorSeq::GetMemoryRequirementBytes() returns SIMD-aligned number of bytes,
    //   so that for each worker its bucket summator is aligned.
    SRBucketSummatorSeq<taNumber>::GetMemoryRequirementBytes() * nWorkers);

  // The shared block for CEEvalQsSubtaskConsider
  const size_t posteriorOffs = runLengthOffs + SRSimd::GetPaddedBytes(sizeof(taNumber) * _dims._nQuestions);
  const size_t threadPosteriorBytes = SRSimd::GetPaddedBytes(sizeof(taNumber) * _dims._nTargets);
  const size_t answerMetricsOffs = posteriorOffs + nWorkers * threadPosteriorBytes;
  const size_t threadAnswerMetricsBytes = 2 * SRSimd::GetPaddedBytes(sizeof(taNumber) * _dims._nAnswers);
  const size_t nWithAnswerMetrics = answerMetricsOffs + nWorkers * threadAnswerMetricsBytes;

  // The shared block for binary search over the run length.
  const size_t grandTotalsOffs = posteriorOffs;
  const size_t nWithGrandTotals = grandTotalsOffs + sizeof(taNumber) * nWorkers;

  const size_t totalBytes = std::max(nWithAnswerMetrics, nWithGrandTotals);
  SRSmartMPP<uint8_t> commonBuf(_memPool, totalBytes);

  SRPoolRunner pr(_tpWorkers, commonBuf.Get() + subtasksOffs);

  TPqaId selQuestion;
  do {
    CEEvalQsTask<taNumber> evalQsTask(*this, *pQuiz, _dims._nTargets - _targetGaps.GetNGaps(),
      commonBuf.Get() + bucketsOffs, SRCast::Ptr<taNumber>(commonBuf.Get() + runLengthOffs),
      commonBuf.Get() + posteriorOffs, threadPosteriorBytes, commonBuf.Get() + answerMetricsOffs,
      threadAnswerMetricsBytes);
    // Although there are no more subtasks which would use this split, it will be used for run-length analysis.
    const SRPoolRunner::Split questionSplit = SRPoolRunner::CalcSplit(commonBuf.Get() + splitOffs, _dims._nQuestions,
      nWorkers);
    {
      SRRWLock<false> rwl(_rws);
      SRPoolRunner::Keeper<CEEvalQsSubtaskConsider<taNumber>> kp = pr.RunPreSplit<CEEvalQsSubtaskConsider<taNumber>>(
        evalQsTask, questionSplit);
    }
    SRAccumulator<taNumber> accTotG(taNumber(0.0));
    const taNumber *const PTR_RESTRICT pRunLength = evalQsTask.GetRunLength();
    taNumber *const PTR_RESTRICT pGrandTotals = SRCast::Ptr<taNumber>(commonBuf.Get() + grandTotalsOffs);
    for (SRThreadCount i = 0; i < questionSplit._nSubtasks; i++) {
      const taNumber curGT = pRunLength[questionSplit._pBounds[i] - 1];
      accTotG.Add(curGT);
      pGrandTotals[i] = accTotG.Get();
    }
    const taNumber totG = pGrandTotals[questionSplit._nSubtasks - 1];
    const taNumber selRunLen = taNumber::MakeRandom(totG, pQuiz->Random());
    const SRThreadCount iWorker = static_cast<SRThreadCount>(
      std::upper_bound(pGrandTotals, pGrandTotals + questionSplit._nSubtasks, selRunLen) - pGrandTotals);
    if (iWorker >= questionSplit._nSubtasks) {
      assert(iWorker == questionSplit._nSubtasks);
      selQuestion = _dims._nQuestions - 1;
      break;
    }
    const TPqaId iFirst = ((iWorker == 0) ? 0 : questionSplit._pBounds[iWorker - 1]);
    const TPqaId iLimit = questionSplit._pBounds[iWorker];
    selQuestion = std::upper_bound(pRunLength + iFirst, pRunLength + iLimit, selRunLen) - pRunLength;
    if (selQuestion >= iLimit) {
      assert(selQuestion == iLimit);
      CELOG(Warning) << SR_FILE_LINE "Hopefully due to a rounding error, within-worker run length binary search hasn't"
        " found a strictly greater value, while the binary search over grand totals pointed to this worker's piece."
        " Random selection: " << selRunLen.ToAmount() << ", worker index " << iWorker << ", grand total "
        << pGrandTotals[iWorker].ToAmount() << ", worker max run length " << pRunLength[iLimit-1].ToAmount();
      selQuestion = iLimit - 1;
    }
  } WHILE_FALSE;

  // If the selected question is in a gap or already answered, try to select the neighboring questions
  if (_questionGaps.IsGap(selQuestion) || SRBitHelper::Test(pQuiz->GetQAsked(), selQuestion)) {
    selQuestion = FindNearestQuestion(selQuestion, *pQuiz);
  }
  if (selQuestion == cInvalidPqaId) {
    err = PqaError(PqaErrorCode::QuestionsExhausted, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Found no unasked"
      " question that is not in a gap."));
    return cInvalidPqaId;
  }
  pQuiz->SetActiveQuestion(selQuestion);
  _nQuestionsAsked.fetch_add(1, std::memory_order_relaxed);
  return selQuestion;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    return PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned("Can't perform regular-only mode"
      " operation (record an answer) because current mode is not regular (but maintenance/shutdown?)."));
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  // Check that iAnswer is within the range
  if (iAnswer < 0 || iAnswer >= _dims._nAnswers) {
    return PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iAnswer, 0, _dims._nAnswers - 1),
      SRString::MakeUnowned("Answer index is not in the answer range."));
  }

  CEQuiz<taNumber> *pQuiz;
  {
    PqaError err;
    pQuiz = UseQuiz(err, iQuiz);
    if (pQuiz == nullptr) {
      assert(!err.IsOk());
      return err;
    }
  }

  return pQuiz->RecordAnswer(iAnswer);
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::ListTopTargets(PqaError& err, const TPqaId iQuiz,
  const TPqaId maxCount, RatedTarget *pDest) 
{
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned("Can't perform regular-only mode"
      " operation (compute next question) because current mode is not regular (but maintenance/shutdown?)."));
    return cInvalidPqaId;
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  CEQuiz<taNumber> *pQuiz = UseQuiz(err, iQuiz);
  if (pQuiz == nullptr) {
    assert(!err.IsOk());
    return cInvalidPqaId;
  }

  const SRThreadCount nWorkers = _tpWorkers.GetWorkerCount();
  const TPqaId nTargets = _dims._nTargets;
  
  //TODO: move this constant to radix sort implementation
  constexpr uint64_t nRadixSortBuckets = 256;
  //TODO: experiment to determine operation weights (comparison vs memory operations).
  const uint64_t nTargPerThread = SRMath::PosDivideRoundUp<uint64_t>(nTargets, nWorkers);
  const uint64_t nRadixSortOps = 9 * std::max<uint64_t>(nTargPerThread, nRadixSortBuckets)
    + uint64_t(maxCount) * std::max(SRMath::CeilLog2(nWorkers), 1ui8);
  const uint64_t nHeapOps = 3 * nTargPerThread + uint64_t(maxCount) * SRMath::CeilLog2(nTargets);
  
  // Currently holds if maxCount > 6 * a / log2(a), where a=nTargets/nWorkers and a>=nRadixSortBuckets
  if (nRadixSortOps < nHeapOps) {
    //TODO: implement and take radix sort branch here
    CELOG(Warning) << "Requested to list " << maxCount << " targets out of " << nTargets << ", which is a large"
      " enough part to prefer radix sort (" << nRadixSortOps << " Ops) over heap (" << nHeapOps << " Ops) approach.";
  }

  // This algorithm is optimized for small number of top targets to list. A separate branch is needed if substantial
  //   part of all targets is to be listed. That branch could use radix sort in separate threads, then heap merge.
  SRMemTotal memTotal;
  //const SRMemItem miSubtasks
  constexpr size_t subtasksOffs = 0;
  const size_t splitOffs = subtasksOffs + nWorkers * SRMaxSizeof</*TODO: make_heap subtask here*/>::value;
  const size_t ratingsOffs = SRSimd::GetPaddedBytes(splitOffs + SRPoolRunner::CalcSplitMemReq(nWorkers));
  const size_t headHeapOffs = SRSimd::GetPaddedBytes(ratingsOffs + nTargets * sizeof(RatedTarget));
  const size_t sourceInfosOffs = headHeapOffs + nWorkers * sizeof(RatingsHeapItem);
  const size_t nWithSourceInfos = sourceInfosOffs + nWorkers * sizeof(RatingsSourceInfo);
  //TODO: radix sort temporary array of ratings, and buckets here
  const size_t totalBytes = nWithSourceInfos;

  SRSmartMPP<uint8_t> commonBuf(_memPool, totalBytes);
  SRPoolRunner pr(_tpWorkers, commonBuf.Get() + subtasksOffs);

  //TODO: implement, assuming normalized priors

  //TODO: remove when implemented
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
  return _nQuestionsAsked.load(std::memory_order_relaxed);
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
