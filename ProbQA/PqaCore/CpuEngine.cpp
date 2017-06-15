#include "stdafx.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/DoubleNumber.h"
#include "../PqaCore/PqaException.h"
#include "../PqaCore/CETrainTask.h"
#include "../PqaCore/InternalErrorHelper.h"

using namespace SRPlat;

namespace ProbQA {

#define CELOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, _pLogger.load(std::memory_order_acquire))

template<typename taNumber> CpuEngine<taNumber>::CpuEngine(const EngineDefinition& engDef)
  : _dims(engDef._dims), _maintSwitch(MaintenanceSwitch::Mode::Regular), _shutdownRequested(0),
  _pLogger(SRDefaultLogger::Get()), _memPool(1 + (engDef._memPoolMaxBytes >> (cLogSimdBits - 3)))
{
  if (_dims._nAnswers < cMinAnswers || _dims._nQuestions < cMinQuestions || _dims._nTargets < cMinTargets)
  {
    throw PqaException(PqaErrorCode::InsufficientEngineDimensions, new InsufficientEngineDimensionsErrorParams(
      _dims._nAnswers, cMinAnswers, _dims._nQuestions, cMinQuestions, _dims._nTargets, cMinTargets));
  }

  taNumber initAmount(engDef._initAmount);
  //// Init cube A: A[ao][q][t] is weight for answer option |ao| for question |q| for target |t|
  _cA.resize(size_t(_dims._nAnswers));
  for (size_t i = 0, iEn= size_t(_dims._nAnswers); i < iEn; i++) {
    _cA[i].resize(size_t(_dims._nQuestions));
    for (size_t j = 0, jEn= size_t(_dims._nQuestions); j < jEn; j++) {
      _cA[i][j].resize(size_t(_dims._nTargets), initAmount);
    }
  }

  //// Init matrix D: D[q][t] is the sum of weigths over all answers for question |q| for target |t|. In the other
  ////   words, D[q][t] is A[0][q][t] + A[1][q][t] + ... + A[K-1][q][t], where K is the number of answer options.
  //// Note that D is subject to summation errors, thus its regular recomputation is desired.
  taNumber initMD = initAmount * _dims._nAnswers;
  _mD.resize(size_t(_dims._nQuestions));
  for (size_t i = 0, iEn=size_t(_dims._nQuestions); i < iEn; i++) {
    _mD[i].resize(size_t(_dims._nTargets), initMD);
  }

  //// Init vector B: the sums of weights over all trainings for each target
  _vB.resize(size_t(_dims._nTargets), initAmount);

  _questionGaps.GrowTo(_dims._nQuestions);
  _targetGaps.GrowTo(_dims._nTargets);

  const uint32_t nWorkers = std::thread::hardware_concurrency();
  for (uint32_t i = 0; i < nWorkers; i++) {
    _workers.emplace_back(&CpuEngine<taNumber>::WorkerEntry, this);
  }
  //throw PqaException(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
  //  "CpuEngine<taNumber>::CpuEngine(const EngineDefinition& engDef)")));
}

template<typename taNumber> CpuEngine<taNumber>::~CpuEngine() {
  PqaError pqaErr = Shutdown();
  if (!pqaErr.isOk() && pqaErr.GetCode() != PqaErrorCode::ObjectShutDown) {
    CELOG(Error) << "Failed CpuEngine::Shutdown(): " << pqaErr.ToString(true);
  }
  for (size_t i = 0, iEn = _stPool.size(); i < iEn; i++) {
    for (size_t j = 0, jEn = _stPool[i].size(); j < jEn; j++) {
      delete _stPool[i][j];
    }
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

  //// Shutdown worker threads
  {
    SRLock<SRCriticalSection> csl(_csWorkers);
    _shutdownRequested = 1;
    _haveWork.WakeAll();
  }
  for (size_t i = 0, iEn = _workers.size(); i < iEn; i++) {
    _workers[i].join();
  }

  return PqaError();
}

template<typename taNumber> CESubtask<taNumber>* CpuEngine<taNumber>::CreateSubtask(
  const typename CESubtask<taNumber>::Kind kind)
{
  switch (kind) {
  case CESubtask<taNumber>::Kind::None:
    CELOG(Critical) << "Requested to create a subtask of kind None";
    return nullptr;
  case CESubtask<taNumber>::Kind::TrainDistrib:
    return new CETrainSubtaskDistrib<taNumber>();
  case CESubtask<taNumber>::Kind::TrainAdd:
    return new CETrainSubtaskAdd<taNumber>();
    //TODO: implement
  default:
    CELOG(Critical) << "Requested to create a subtask of unhandled kind #" + static_cast<int64_t>(kind);
    return nullptr;
  }
}

template<typename taNumber> void CpuEngine<taNumber>::DeleteSubtask(CESubtask<taNumber> *pSubtask) {
  delete pSubtask;
}

template<typename taNumber> CESubtask<taNumber>* CpuEngine<taNumber>::AcquireSubtask(
  const typename CESubtask<taNumber>::Kind kind)
{
  SRLock<TStpSync> stpsl(_stpSync);
  size_t iKind = static_cast<size_t>(kind);
  if (iKind >= _stPool.size() || _stPool[iKind].size() == 0) {
    return CreateSubtask(kind);
  }
  CESubtask<taNumber> *answer = _stPool[iKind].back();
  _stPool[iKind].pop_back();
  return answer;
}

template<typename taNumber> void CpuEngine<taNumber>::ReleaseSubtask(CESubtask<taNumber> *pSubtask) {
  // With a subtask pool, avoid memory alloc/free bottlenecks
  SRLock<TStpSync> stpsl(_stpSync);
  size_t kind = static_cast<size_t>(pSubtask->GetKind());
  size_t poolSize = _stPool.size();
  if (kind >= poolSize) {
    _stPool.resize(kind + 1);
  }
  _stPool[kind].push_back(pSubtask);
}

template<typename taNumber> void CpuEngine<taNumber>::RunSubtask(CESubtask<taNumber> &ceSt) {
  switch (ceSt.GetKind()) {
  case CESubtask<taNumber>::Kind::None:
    CELOG(Critical) << "Worker has received a subtask of kind None";
    break;
  case CESubtask<taNumber>::Kind::TrainDistrib:
    RunTrainDistrib(dynamic_cast<CETrainSubtaskDistrib<taNumber>&>(ceSt));
    break;
  case CESubtask<taNumber>::Kind::TrainAdd:
    RunTrainAdd(dynamic_cast<CETrainSubtaskAdd<taNumber>&>(ceSt));
    //TODO: implement
  default:
    CELOG(Critical) << "Worker has received a subtask of unhandled kind #" + static_cast<int64_t>(ceSt.GetKind());
    break;
  }
}

template<typename taNumber> void CpuEngine<taNumber>::WorkerEntry() {
  for (;;) {
    CESubtaskCompleter<taNumber> ceStc;
    try {
      {
        SRLock<SRCriticalSection> csl(_csWorkers);
        while (_quWork.size() == 0) {
          if (_shutdownRequested) {
            return;
          }
          _haveWork.Wait(_csWorkers);
        }
        ceStc.Set(_quWork.front());
        _quWork.pop();
      }
      RunSubtask(*ceStc.Get());
    }
    catch (SRException& ex) {
      PqaError err;
      err.SetFromException(ex);
      CELOG(Critical) << "Worker thread got an SRException not handled at lower levels: " << err.ToString(true);
      ceStc.Get()->_pTask->AddError(err);
    }
    catch (std::exception& ex) {
      PqaError err;
      err.SetFromException(ex);
      CELOG(Critical) << "Worker thread got an std::exception not handled at lower levels: " << err.ToString(true);
      ceStc.Get()->_pTask->AddError(err);
    }
  }
}

template<typename taNumber> void CpuEngine<taNumber>::WakeWorkersWait(CETask<taNumber> &task) {
  _haveWork.WakeAll();
  SRLock<SRCriticalSection> csl(_csWorkers);
  for (;;) {
    task._isComplete.Wait(_csWorkers);
    if (task.GetToDo() == 0) {
      break;
    }
  }
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

template<typename taNumber> void CpuEngine<taNumber>::RunTrainAdd(CETrainSubtaskAdd<taNumber> &tsa) {

}

template<typename taNumber> PqaError CpuEngine<taNumber>::Train(const TPqaId nQuestions,
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
  const size_t nWorkers = _workers.size();
  //// Allocate memory out of locks
  SRSmartMPP<TMemPool, TPqaId> ttPrev(_memPool, nQuestions);
  SRSmartMPP<TMemPool, std::atomic<TPqaId>> ttLast(_memPool, nWorkers);

  CETrainTask<taNumber> trainTask(this, amount);
  trainTask._prev = ttPrev.Get();
  trainTask._last = ttLast.Get();
  //TODO: vectorize/parallelize
  for (size_t i = 0; i < nWorkers; i++) {
    new(trainTask._last+i) std::atomic<TPqaId>(cInvalidPqaId);
  }
  // &trainTask, &nWorkers
  auto&& ttLastFinally = SRMakeFinally([&pLast = trainTask._last, &nWorkers] {
    //TODO: vectorize/parallelize
    for (size_t i = 0; i < nWorkers; i++) {
      pLast[i].~atomic();
    }
  }); (void)ttLastFinally; // prevent warning C4189

  { // Scope for the locks
    MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
    SRRWLock<true> rwl(_rws);

    //// This code must be reader-writer locked, because we are validating the input before modifying the KB, so noone
    ////   must change the KB in between.

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

    { //// Distribute the AQs into buckets with the number of buckets divisable by the number of workers.
      lldiv_t perWorker = div((long long)nQuestions, (long long)nWorkers);
      TPqaId nextStart = 0;
      SRLock<SRCriticalSection> csl(_csWorkers);
      size_t i = 0;
      for (; i < nWorkers && nextStart < nQuestions; i++) {
        TPqaId curStart = nextStart;
        nextStart += perWorker.quot;
        if ((long long)i < perWorker.rem) {
          nextStart++;
        }
        assert(nextStart <= nQuestions);
        auto pSt = AcquireSubtask(CESubtask<taNumber>::Kind::TrainDistrib);
        auto pTsd = dynamic_cast<CETrainSubtaskDistrib<taNumber>*>(pSt);
        if (pTsd == nullptr) {
          // Handle and report error. The problem is that some subtasks have been already pushed to the queue, and
          //  they have a pointer to an object on the stack of the current function.
          const char* const msg = "Internal error: acquired something other than CETrainSubtaskDistrib for code"
            " TrainDistrib .";
          CELOG(Critical) << msg;
          resErr = MAKE_INTERR_MSG(SRString::MakeUnowned(msg));
          trainTask.Cancel();
          // Don't return a broken subtask to the pool.
          DeleteSubtask(pSt);
          break;
        }
        pTsd->_pTask = &trainTask;
        pTsd->_pFirst = pAQs + curStart;
        pTsd->_pLim = pAQs + nextStart;
        _quWork.push(pTsd);
      }
      trainTask.IncToDo(i);
    }
    // Even if the task has been cancelled, wait till all the workers acknowledge that
    WakeWorkersWait(trainTask);
    if (!resErr.isOk()) {
      return resErr;
    }
    resErr = trainTask.TakeAggregateError(SRString::MakeUnowned("Failed validating and bucketing the input."));
    if (!resErr.isOk()) {
      return resErr;
    }
    // Phase 1 complete

    trainTask.PrepareNextPhase();

    // Phase 2: update KB
    for (size_t i = 0; i < nWorkers; i++) {
      auto pSt = AcquireSubtask(CESubtask<taNumber>::Kind::TrainAdd);
      auto pTsa = dynamic_cast<CETrainSubtaskAdd<taNumber>*>(pSt);
      if (pTsa == nullptr) {
        // Handle and report error. The problem is that some subtasks have been already pushed to the queue, and
        //  they have a pointer to an object on the stack of the current function.
        const char* const msg = "Internal error: acquired something other than CETrainSubtaskAdd for code"
          " TrainAdd .";
        CELOG(Critical) << msg;
        resErr = MAKE_INTERR_MSG(SRString::MakeUnowned(msg));
        trainTask.Cancel();
        // Don't return a broken subtask to the pool.
        DeleteSubtask(pSt);
        break;
      }
    }
    trainTask.IncToDo(nWorkers);
    // Even if the task has been cancelled, wait till all the workers acknowledge that
    WakeWorkersWait(trainTask);
    
    if (!resErr.isOk()) {
      return resErr;
    }
    resErr = trainTask.TakeAggregateError(SRString::MakeUnowned("Failed adding to KB the given training data."));
    if (!resErr.isOk()) {
      return resErr;
    }
    
    //TODO: update target totals |_vB|

    // This method should increase the counter of questions asked by the number of questions in this training.
    _nQuestionsAsked += nQuestions;
  }
  
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::Train")));
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::StartQuiz(PqaError& err) {
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::StartQuiz")));
  return cInvalidPqaId;
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::ResumeQuiz(PqaError& err, const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs) 
{
  (void)nQuestions; (void)pAQs; //TODO: remove when implemented
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::ResumeQuiz")));
  return cInvalidPqaId;
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

template<typename taNumber> TPqaId CpuEngine<taNumber>::AddQuestion(PqaError& err, const TPqaAmount initialAmount) {
  (void)initialAmount; //TODO: remove when implemented
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddQuestion")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::AddQuestions(TPqaId nQuestions, AddQuestionParam *pAqps) {
  (void)nQuestions; (void)pAqps; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddQuestions")));
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::AddTarget(PqaError& err, const TPqaAmount initialAmount) {
  (void)initialAmount; //TODO: remove when implemented
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddTarget")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::AddTargets(TPqaId nTargets, AddTargetParam *pAtps) {
  (void)nTargets; (void)pAtps; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddTargets")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RemoveQuestion(const TPqaId iQuestion) {
  (void)iQuestion; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RemoveQuestion")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds)
{
  (void)nQuestions; (void)pQIds; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RemoveQuestions")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RemoveTarget(const TPqaId iTarget) {
  (void)iTarget; //TODO: remove when implemented
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RemoveTarget")));
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