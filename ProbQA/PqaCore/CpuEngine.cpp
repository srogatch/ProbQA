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

template<typename taNumber> SRThreadPool::TThreadCount CpuEngine<taNumber>::CalcCompThreads() {
  return std::thread::hardware_concurrency();
}

template<typename taNumber> SRThreadPool::TThreadCount CpuEngine<taNumber>::CalcMemOpThreads()
{
  // This is a trivial heuristic based on the observation that on Ryzen 1800X with 2 DDR4 modules in a single memory
  //   channel, the maximum copy speed is achieved for 5 threads.
  return std::max(1ui32, std::min(CalcCompThreads(), 5ui32));
}

template<typename taNumber> CpuEngine<taNumber>::CpuEngine(const EngineDefinition& engDef)
  : _dims(engDef._dims), _maintSwitch(MaintenanceSwitch::Mode::Regular), _shutdownRequested(0),
  _pLogger(SRDefaultLogger::Get()), _memPool(1 + (engDef._memPoolMaxBytes >> SRSimd::_cLogNBytes)),
  _tpWorkers(CalcCompThreads()), _nMemOpThreads(CalcMemOpThreads())
{
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

template<typename taNumber> template<typename taSubtask, typename taCallback> PqaError
CpuEngine<taNumber>::SplitAndRunSubtasks(const size_t nWorkers, CETask<taNumber> &task, const size_t nItems,
  void *pSubtaskMem, const taCallback &onVisit)
{
  taSubtask *pSubtasks = reinterpret_cast<taSubtask*>(pSubtaskMem);
  size_t nSubtasks = 0;
  bool bWorkersFinished = false;
  size_t nextStart = 0;
  lldiv_t perWorker = div((long long)nItems, (long long)nWorkers);

  auto&& subtasksFinally = SRMakeFinally([&] {
    if (!bWorkersFinished) {
      task.WaitComplete();
    }
    for (size_t i = 0; i < nSubtasks; i++) {
      pSubtasks[i].~taSubtask<taNumber>();
    }
  }); (void)subtasksFinally;

  while (nSubtasks < nWorkers && nextStart < nQuestions) {
    size_t curStart = nextStart;
    nextStart += perWorker.quot;
    if ((long long)nSubtasks < perWorker.rem) {
      nextStart++;
    }
    assert(nextStart <= nQuestions);
    taCallback(pSubtasks + nSubtasks, curStart, nextStart);
    // For finalization, it's important to increment subtask counter right after another subtask has been
    //   constructed.
    nSubtasks++;
    _tpWorkers.Enqueue(pSubtasks + nSubtasks - 1);
  }

  bWorkersFinished = true; // Don't call again SRBaseTask::WaitComplete() if it throws here.
  task.WaitComplete();

  return task.TakeAggregateError(SRString::MakeUnowned("Failed validating and bucketing the input."));
}

template<typename taNumber> void CpuEngine<taNumber>::RunTrainDistrib(CETrainSubtaskDistrib<taNumber> &tsd) {
  CETrainTask<taNumber> *pTask = static_cast<CETrainTask<taNumber>*>(tsd._pTask);
  if (pTask->IsCancelled()) {
    return;
  }
  for (const AnsweredQuestion *pAQ = tsd._pFirst, *pEn= tsd._pLim; pAQ < pEn; pAQ++) {
    // Check ranges
    const TPqaId iQuestion = pAQ->_iQuestion;
    if (iQuestion < 0 || iQuestion >= _dims._nQuestions) {
      const TPqaId nKB = _dims._nQuestions;
      pTask->AddError(PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iQuestion, 0, nKB - 1),
        SRString::MakeUnowned("Question index is not in KB range.")));
      return;
    }
    if (_questionGaps.IsGap(iQuestion)) {
      pTask->AddError(PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iQuestion), SRString::MakeUnowned(
        "Question index is not in KB (but rather at a gap).")));
      return;
    }
    const TPqaId iAnswer = pAQ->_iAnswer;
    if (iAnswer < 0 || iAnswer >= _dims._nAnswers) {
      const TPqaId nKB = _dims._nAnswers;
      pTask->AddError(PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iAnswer, 0, nKB - 1),
        SRString::MakeUnowned("Answer index is not in KB range.")));
      return;
    }
    // Sort questions into buckets so that workers in the next phase do not race for data.
    const TPqaId iBucket = iQuestion % _workers.size();
    const TPqaId iPrev = pTask->_iPrev.fetch_add(1);
    TPqaId expected = pTask->_last[iBucket].load(std::memory_order_acquire);
    while (!pTask->_last[iBucket].compare_exchange_weak(expected, iPrev, std::memory_order_acq_rel,
      std::memory_order_acquire));
    pTask->_prev[iPrev] = expected;
  }
}

template<> void CpuEngine<DoubleNumber>::RunTrainAdd(CETrainSubtaskAdd<DoubleNumber> &tsa) {
  CETrainTask<DoubleNumber> &task = static_cast<CETrainTask<DoubleNumber>&>(*tsa._pTask);
  const CETrainTask<DoubleNumber>& cTask = task; // enable optimizations with const
  TPqaId iLast = cTask._last[tsa._iWorker];
  if (iLast == cInvalidPqaId) {
    return;
  }
  const TPqaId *const cPrev = cTask._prev;
  // Enable optimizations with const
  const __m256d& fullAddend = cTask._numSpec._fullAddend;
  const __m256d& collAddend = cTask._numSpec._collAddend;
  do {
    const AnsweredQuestion& aqFirst = cTask._pAQs[iLast];
    iLast = cPrev[iLast];
    if (iLast == cInvalidPqaId) {
      // Use SSE2 instead of AVX here to supposedly reduce the load on the CPU core (better hyperthreading).
      __m128d sum = _mm_set_pd(
        _mD[SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)].GetValue(),
        _sA[SRCast::ToSizeT(aqFirst._iAnswer)][SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
          .GetValue());
      sum = _mm_add_pd(sum, *reinterpret_cast<const __m128d*>(&fullAddend));
      _sA[SRCast::ToSizeT(aqFirst._iAnswer)][SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
        .SetValue(sum.m128d_f64[0]);
      _mD[SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)].SetValue(sum.m128d_f64[1]);
      return;
    }
    const AnsweredQuestion& aqSecond = cTask._pAQs[iLast];
    if (aqFirst._iQuestion == aqSecond._iQuestion) {
      // Vectorize 3 additions, with twice the amount in element #1
      __m256d sum = _mm256_set_pd(0,
        _sA[SRCast::ToSizeT(aqSecond._iAnswer)][SRCast::ToSizeT(aqSecond._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
          .GetValue(),
        _mD[SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)].GetValue(),
        _sA[SRCast::ToSizeT(aqFirst._iAnswer)][SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
          .GetValue());
      sum = _mm256_add_pd(sum, collAddend);
      _sA[SRCast::ToSizeT(aqFirst._iAnswer)][SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
        .SetValue(sum.m256d_f64[0]);
      _mD[SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)].SetValue(sum.m256d_f64[1]);
      _sA[SRCast::ToSizeT(aqSecond._iAnswer)][SRCast::ToSizeT(aqSecond._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
        .SetValue(sum.m256d_f64[2]);
    }
    else {
      // Finally we can vectorize all the 4 additions
      __m256d sum = _mm256_set_pd(
        _mD[SRCast::ToSizeT(aqSecond._iQuestion)][SRCast::ToSizeT(cTask._iTarget)].GetValue(),
        _sA[SRCast::ToSizeT(aqSecond._iAnswer)][SRCast::ToSizeT(aqSecond._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
          .GetValue(),
        _mD[SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)].GetValue(),
        _sA[SRCast::ToSizeT(aqFirst._iAnswer)][SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
          .GetValue());
      sum = _mm256_add_pd(sum, fullAddend);
      _sA[SRCast::ToSizeT(aqFirst._iAnswer)][SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
        .SetValue(sum.m256d_f64[0]);
      _mD[SRCast::ToSizeT(aqFirst._iQuestion)][SRCast::ToSizeT(cTask._iTarget)].SetValue(sum.m256d_f64[1]);
      _sA[SRCast::ToSizeT(aqSecond._iAnswer)][SRCast::ToSizeT(aqSecond._iQuestion)][SRCast::ToSizeT(cTask._iTarget)]
        .SetValue(sum.m256d_f64[2]);
      _mD[SRCast::ToSizeT(aqSecond._iQuestion)][SRCast::ToSizeT(cTask._iTarget)].SetValue(sum.m256d_f64[3]);
    }
    iLast = cPrev[iLast];
  } while (iLast != cInvalidPqaId);
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
  const size_t nWorkers = _tpWorkers.GetWorkerCount();
  //// Do a single allocation for all needs. Allocate memory out of locks.
  // For proper alignment, the data must be laid out in the decreasing order of item alignments.
  const size_t ttLastOffs = std::max(sizeof(CETrainSubtaskDistrib<taNumber>), sizeof(CETrainSubtaskAdd<taNumber>))
    * nWorkers;
  const size_t ttPrevOffs = ttLastOffs + sizeof(std::atomic<TPqaId>) * nWorkers;
  const size_t nTotalBytes = ttPrevOffs + sizeof(TPqaId) * SRCast::ToSizeT(nQuestions);
  SRSmartMPP<TMemPool, uint8_t> commonBuf(_memPool, nTotalBytes);

  CETrainTask<taNumber> trainTask(this, iTarget, pAQs);
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

    SplitAndRunSubtasks<CETrainSubtaskDistrib<taNumber>>(nWorkers, trainTask, nQuestions, commonBuf.Get(),
      [&](CETrainSubtaskDistrib<taNumber> *pCurSt, const size_t curStart, const size_t nextStart)
    {
      new (pCurSt) CETrainSubtaskDistrib<taNumber>(&trainTask, pAQs + curStart, pAQs + nextStart);
    });
    { //// Distribute the AQs into buckets with the number of buckets divisable by the number of workers.
      CETrainSubtaskDistrib<taNumber> *pSubtasks = reinterpret_cast<CETrainSubtaskDistrib<taNumber>*>(commonBuf.Get());
      size_t nSubtasks = 0;
      bool bWorkersFinished = false;
      TPqaId nextStart = 0;
      lldiv_t perWorker = div((long long)nQuestions, (long long)nWorkers);

      auto&& subtasksFinally = SRMakeFinally([&] {
        if (!bWorkersFinished) {
          trainTask.WaitComplete();
        }
        for (size_t i = 0; i < nSubtasks; i++) {
          pSubtasks[i].~CETrainSubtaskDistrib<taNumber>();
        }
      }); (void)subtasksFinally;

      while (nSubtasks < nWorkers && nextStart < nQuestions) {
        TPqaId curStart = nextStart;
        nextStart += perWorker.quot;
        if ((long long)nSubtasks < perWorker.rem) {
          nextStart++;
        }
        assert(nextStart <= nQuestions);
        new (pSubtasks + nSubtasks) CETrainSubtaskDistrib<taNumber>(&trainTask, pAQs + curStart, pAQs + nextStart);
        // For finalization, it's important to increment subtask counter right after another subtask has been
        //   constructed.
        nSubtasks++;
        _tpWorkers.Enqueue(pSubtasks + nSubtasks - 1);
      }

      bWorkersFinished = true; // Don't call again SRBaseTask::WaitComplete() if it throws here.
      trainTask.WaitComplete();

      resErr = trainTask.TakeAggregateError(SRString::MakeUnowned("Failed validating and bucketing the input."));
      if (!resErr.isOk()) {
        return resErr;
      }
    } // Phase 1 complete

    { // Phase 2: update KB
      CETrainSubtaskAdd<taNumber> *pSubtasks = reinterpret_cast<CETrainSubtaskAdd<taNumber>*>(commonBuf.Get());
      size_t nSubtasks = 0;
      bool bWorkersFinished = false;

      auto&& subtasksFinally = SRMakeFinally([&] {
        if (!bWorkersFinished) {
          trainTask.WaitComplete();
        }
        for (size_t i = 0; i < nSubtasks; i++) {
          pSubtasks[i].~CETrainSubtaskAdd<taNumber>();
        }
      }); (void)subtasksFinally;

      while (nSubtasks < nWorkers) {
        //TODO: implement
      }
    }

    {
      SRLock<SRCriticalSection> csl(_csWorkers);
      size_t i = 0;
      for (; i < nWorkers; i++) {
        auto pTsa = AcquireSubtask<CETrainSubtaskAdd<taNumber>>();
        if (pTsa == nullptr) {
          // Handle and report error. The problem is that some subtasks have been already pushed to the queue, and
          //  they have a pointer to an object on the stack of the current function.
          const char* const msg = "Internal error: failed to acquire CETrainSubtaskAdd at " SR_FILE_LINE;
          CELOG(Critical) << msg;
          resErr = MAKE_INTERR_MSG(SRString::MakeUnowned(msg));
          trainTask.Cancel();
          break;
        }
        pTsa->_pTask = &trainTask;
        pTsa->_iWorker = static_cast<decltype(pTsa->_iWorker)>(i);
        _quWork.push(pTsa);
      }
      trainTask.IncToDo(i);
    }
    // Even if the task has been cancelled, wait till all the workers acknowledge that
    WakeWorkersWait(trainTask);

    if (!resErr.isOk()) {
      return resErr;
    }
    resErr = trainTask.TakeAggregateError(SRString::MakeUnowned("Failed adding to KB the given training data."));
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
    const auto msMode = MaintenanceSwitch::Mode::Regular;
    if (!_maintSwitch.TryEnterSpecific<msMode>()) {
      op._err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned("Can't perform regular-only mode"
        " operation because current mode is not regular (but maintenance/shutdown?)."));
      return cInvalidPqaId;
    }
    MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);
    // So long as this constructor only needs the number of questions and targets, it can be out of _rws because
    //   _maintSwitch guards engine dimensions.
    const size_t nTargets = SRCast::ToSizeT(_dims._nTargets);
    const size_t nQuestions = SRCast::ToSizeT(_dims._nQuestions);
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
    {
      SRRWLock<false> rwl(_rws);

      //// Copy prior likelihoods for targets
      //TODO: launch these 3 memory operations in separate threads
      SRUtils::Copy256<false, false>(op._pQuiz->GetTlhMants(), _vB.Get(),
        SRSimd::VectsFromBytes(nTargets * sizeof(taNumber)));
      SRUtils::FillZeroVects<false>(reinterpret_cast<__m256i*>(op._pQuiz->GetTlhExps()),
        SRSimd::VectsFromBytes(nTargets * sizeof(CEQuiz<taNumber>::TExponent)));
      SRUtils::FillZeroVects<true>(op._pQuiz->GetQAsked(), SRSimd::VectsFromBits(nQuestions));

      if (op._err.isOk()) {
        op.MaybeUpdatePriorsWithAnsweredQuestions(this);
      }
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

  const size_t nWorkers = _workers.size();
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