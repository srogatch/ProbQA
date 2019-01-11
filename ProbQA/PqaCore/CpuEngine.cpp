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
#include "../PqaCore/CEListTopTargetsAlgorithm.h"
#include "../PqaCore/CETrainOperation.h"
#include "../PqaCore/TargetRowPersistence.h"

using namespace SRPlat;

namespace ProbQA {

#define CELOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, _pLogger.load(std::memory_order_acquire))

template<typename taNumber> size_t CpuEngine<taNumber>::CalcWorkerStackSize(const EngineDimensions& dims) {
  const size_t szNextQuestion = std::max({CEEvalQsSubtaskConsider<taNumber>::CalcStackReq(dims)});

  return std::max({szNextQuestion});
}

template<typename taNumber> CpuEngine<taNumber>::CpuEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi)
  : BaseCpuEngine(engDef, CalcWorkerStackSize(engDef._dims), pKbFi)
{
  const size_t nQuestions = SRCast::ToSizeT(_dims._nQuestions);
  const size_t nAnswers = SRCast::ToSizeT(_dims._nAnswers);
  const size_t nTargets = SRCast::ToSizeT(_dims._nTargets);

  TargetRowPersistence<taNumber> trp = ((pKbFi == nullptr) ? TargetRowPersistence<taNumber>()
    : TargetRowPersistence<taNumber>(pKbFi->_sf, nTargets));

  const taNumber init1(engDef._initAmount);
  const taNumber initSqr = taNumber(init1).Sqr();
  //// Init cube A: A[q][ao][t] is weight for answer option |ao| for question |q| for target |t|
  _sA.resize(nQuestions);
  for (size_t i = 0, iEn= nQuestions; i < iEn; i++) {
    _sA[i].resize(SRCast::ToSizeT(_dims._nAnswers));
    for (size_t k = 0, kEn= nAnswers; k < kEn; k++) {
      _sA[i][k].Resize<false>(nTargets);
      if (pKbFi == nullptr) {
        _sA[i][k].FillAll<false>(initSqr);
        continue;
      }
      if(!trp.Read(_sA[i][k])) {
        PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRMessageBuilder(SR_FILE_LINE
          "Can't read the target dimension of _sA weights at [")(i)(", ")(k)("].").GetOwnedSRString()).ThrowMoving();
      }
    }
  }

  //// Init matrix D: D[q][t] is the sum of weigths over all answers for question |q| for target |t|. In the other
  ////   words, D[q][t] is A[q][0][t] + A[q][1][t] + ... + A[q][K-1][t], where K is the number of answer options.
  //// Note that D is subject to summation errors, thus its regular recomputation is desired.
  const taNumber initMD = initSqr * nAnswers;
  _mD.resize(size_t(_dims._nQuestions));
  for (size_t i = 0, iEn=nQuestions; i < iEn; i++) {
    _mD[i].Resize<false>(nTargets);
    if (pKbFi == nullptr) {
      _mD[i].FillAll<false>(initMD);
      continue;
    }
    if(!trp.Read(_mD[i])) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRMessageBuilder(SR_FILE_LINE
        "Can't read the target dimension of _mD weights at [")(i)("].").GetOwnedSRString()).ThrowMoving();
    }
  }

  //// Init vector B: the sums of weights over all trainings for each target
  _vB.Resize<false>(nTargets);
  if (pKbFi == nullptr) {
    _vB.FillAll<false>(init1);
  }
  else {
    if(!trp.Read(_vB)) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(SR_FILE_LINE
        "Can't read the _vB weights.")).ThrowMoving();
    }
  }

  AfterStatisticsInit(pKbFi);
}

template<typename taNumber> CpuEngine<taNumber>::~CpuEngine() {
  PqaError pqaErr = Shutdown();
  if (!pqaErr.IsOk() && pqaErr.GetCode() != PqaErrorCode::ObjectShutDown) {
    CELOG(Error) << "Failed CpuEngine::Shutdown(): " << pqaErr.ToString(true);
  }
}

template<typename taNumber> PqaError CpuEngine<taNumber>::TrainSpec(const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs, const TPqaId iTarget, const TPqaAmount amount)
{
  PqaError resErr;
  const SRThreadCount nWorkers = _tpWorkers.GetWorkerCount();
  //// Do a single allocation for all needs. Allocate memory out of locks.
  // For proper alignment, the data must be laid out in the decreasing order of item alignments.
  SRMemTotal mtCommon;
  const SRByteMem miSubtasks(nWorkers * SRMaxSizeof<CETrainSubtaskDistrib<taNumber>,
    CETrainSubtaskAdd<taNumber> >::value, SRMemPadding::None, mtCommon);
  const SRMemItem<std::atomic<TPqaId>> miTtLast(nWorkers, SRMemPadding::None, mtCommon);
  const SRMemItem<TPqaId> miTtPrev(SRCast::ToSizeT(nQuestions), SRMemPadding::None, mtCommon);
  SRSmartMPP<uint8_t> commonBuf(_memPool, mtCommon._nBytes);

  CETrainTask<taNumber> trainTask(*this, nWorkers, iTarget, pAQs, amount);
  //TODO: these are slow because threads share a cache line. It's not clear yet how to workaround this: the data is not
  //  per-source thread, but rather per target thread (after distribution).
  trainTask._prev = miTtPrev.Ptr(commonBuf);
  trainTask._last = miTtLast.Ptr(commonBuf);
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
      return PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iTarget), SRString::MakeUnowned(SR_FILE_LINE
        "Target index is not in KB (but rather at a gap)."));
    }

    SRPoolRunner pr(_tpWorkers, miSubtasks.BytePtr(commonBuf));

    //// Distribute the AQs into buckets with the number of buckets divisable by the number of workers.
    pr.SplitAndRunSubtasks<CETrainSubtaskDistrib<taNumber>>(trainTask, nQuestions, trainTask.GetWorkerCount(),
      [&](void *pStMem, SRSubtaskCount iWorker, int64_t iFirst, int64_t iLimit) {
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

    //TODO: why is this inside the locks?
    // This method should increase the counter of questions asked by the number of questions in this training.
    _nQuestionsAsked.fetch_add(nQuestions, std::memory_order_relaxed);
  }

  return PqaError();
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::CreateQuizInternal(CECreateQuizOpBase &op) {
  try {
    struct NoSrwTask : public CETask {
      CEQuiz<taNumber> *_pQuiz;
    public: // methods
      explicit NoSrwTask(CpuEngine<taNumber> &ce) : CETask(ce, /*nWorkers*/ 3) { }
    } tNoSrw(*this); // Subtasks without SRW locked
    tNoSrw.Reset();

    constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
    if (!_maintSwitch.TryEnterSpecific<msMode>()) {
      op._err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform"
        " regular-only mode operation (Start/Resume quiz) because current mode is not regular"
        " (but maintenance/shutdown?)."));
      return cInvalidPqaId;
    }
    MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

    // So long as this constructor only needs the number of questions and targets, it can be out of _rws because
    //   _maintSwitch guards engine dimensions in Regular mode (but not in Maintenance mode).
    SRObjectMPP<CEQuiz<taNumber>> spQuiz(_memPool, this);
    tNoSrw._pQuiz = spQuiz.Get();
    const TPqaId quizId = AssignQuiz(spQuiz.Get());
    try {
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
            for (size_t i = 0; i < SRCast::ToSizeT(resumeOp._nAnswered); i++) {
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

        if (op.IsResume()) {
          SRTaskWaiter noSrwTaskWaiter(&tNoSrw);
          _tpWorkers.Enqueue({ &lstSetQAsked, &lstAddAnswers }, tNoSrw);
        }
        else {
          // Run in the current thread
          lstSetQAsked.Run();
        }
      }
      op._err = tNoSrw.TakeAggregateError();
      if (op._err.IsOk()) {
        // If it's "resume quiz" operation, update the prior likelihoods with the questions answered, and normalize the
        //   priors. If it's "start quiz" operation, just divide the priors by their sum.
        op.UpdateLikelihoods(*this, *spQuiz.Get());
      }
    }
    CATCH_TO_ERR_SET(op._err);
    if (!op._err.IsOk()) {
      spQuiz.EarlyRelease();
      UnassignQuiz(quizId);
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

template<typename taNumber> TPqaId CpuEngine<taNumber>::ResumeQuizSpec(PqaError& err, const TPqaId nAnswered,
  const AnsweredQuestion* const pAQs) 
{
  CECreateQuizResume<taNumber> resumeOp(err, nAnswered, pAQs);
  return CreateQuizInternal(resumeOp);
}

template<typename taNumber> PqaError CpuEngine<taNumber>::NormalizePriors(CEQuiz<taNumber> &quiz, SRPoolRunner &pr,
  const SRPoolRunner::Split& targSplit)
{
  CENormPriorsTask<taNumber> normPriorsTask(*this, quiz);

  { // The lifetime for maximum selection subtasks
    SRPoolRunner::Keeper<CENormPriorsSubtaskMax<taNumber>> kp = pr.RunPreSplit<CENormPriorsSubtaskMax<taNumber>>(
      normPriorsTask, targSplit);
    assert(kp.GetNSubtasks() == targSplit._nSubtasks);
    const SRSubtaskCount nResultVects = kp.GetNSubtasks() >> SRSimd::_cLogNComps64;
    const SRVectCompCount nTail = kp.GetNSubtasks() & ((SRSubtaskCount(1) << SRSimd::_cLogNComps64) - 1);
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
      for (SRSubtaskCount i = 1; i < nResultVects; i++) {
        const SRSubtaskCount iBase = (i << SRSimd::_cLogNComps64);
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

  { // Correct the exponents towards the taNumber range, and calculate their sum
    typedef CENormPriorsSubtaskCorrSum<taNumber> TCorrSumSubtask;
    SRPoolRunner::Keeper<TCorrSumSubtask> kp = pr.RunPreSplit<TCorrSumSubtask>(normPriorsTask, targSplit);
    Summator<taNumber>::ForPriors(kp, normPriorsTask);
  }

  // Divide priors by their sum, so to get probabilities.
  pr.RunPreSplit<CEDivTargPriorsSubtask<CENormPriorsTask<taNumber>>>(normPriorsTask, targSplit);

  return PqaError();
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::NextQuestionSpec(PqaError& err, BaseQuiz *pBaseQuiz) {
  CEQuiz<taNumber> *pQuiz = static_cast<CEQuiz<taNumber>*>(pBaseQuiz);
  const SRSubtaskCount nWorkers = _tpWorkers.GetWorkerCount() * 8;
  SRMemTotal mtCommon;
  const SRByteMem miSubtasks(nWorkers * SRMaxSizeof<CEEvalQsSubtaskConsider<taNumber> >::value, SRMemPadding::None,
    mtCommon);
  const SRByteMem miSplit(SRPoolRunner::CalcSplitMemReq(nWorkers), SRMemPadding::Both, mtCommon);
  const SRMemItem<taNumber> miRunLength(_dims._nQuestions, SRMemPadding::Both, mtCommon);
  const SRMemItem<taNumber> miGrandTotals(nWorkers, SRMemPadding::Both, mtCommon);

  SRSmartMPP<uint8_t> commonBuf(_memPool, mtCommon._nBytes);
  SRPoolRunner pr(_tpWorkers, miSubtasks.BytePtr(commonBuf));

  TPqaId selQuestion;
  do {
    CEEvalQsTask<taNumber> evalQsTask(*this, *pQuiz, _dims._nTargets - _targetGaps.GetNGaps(),
      miRunLength.Ptr(commonBuf));
    // Although there are no more subtasks which would use this split, it will be used for run-length analysis.
    const SRPoolRunner::Split questionSplit = SRPoolRunner::CalcSplit(miSplit.BytePtr(commonBuf), _dims._nQuestions,
      nWorkers);
    {
      SRRWLock<false> rwl(_rws);
      SRPoolRunner::Keeper<CEEvalQsSubtaskConsider<taNumber>> kp = pr.RunPreSplit<CEEvalQsSubtaskConsider<taNumber>>(
        evalQsTask, questionSplit);
    }
    SRAccumulator<taNumber> accTotG(taNumber(0.0));
    const taNumber *const PTR_RESTRICT pRunLength = evalQsTask.GetRunLength();
    taNumber *const PTR_RESTRICT pGrandTotals = miGrandTotals.Ptr(commonBuf);
    for (SRSubtaskCount i = 0; i < questionSplit._nSubtasks; i++) {
      const taNumber curGT = pRunLength[questionSplit._pBounds[i] - 1];
      accTotG.Add(curGT);
      pGrandTotals[i] = accTotG.Get();
      //TODO: for performance reasons this check should be moved to subtasks, but here it checks consistency better
      if (!pGrandTotals[i].IsFinite()) {
        CELOG(Error) << SR_FILE_LINE << "Overflow or underflow has happened in the question evaluation subtasks: "
          << pGrandTotals[i].ToAmount();
      }
    }
    const taNumber totG = pGrandTotals[questionSplit._nSubtasks - 1];
    if (totG <= TPqaAmount(0)) {
      CELOG(Warning) << SR_FILE_LINE << "Grand-grand total is " << totG.ToAmount();
    }
    const taNumber selRunLen = taNumber::MakeRandom(totG, SRFastRandom::ThreadLocal());
    const SRSubtaskCount iWorker = static_cast<SRSubtaskCount>(
      std::upper_bound(pGrandTotals, pGrandTotals + questionSplit._nSubtasks, selRunLen) - pGrandTotals);
    if (iWorker >= questionSplit._nSubtasks) {
      assert(iWorker == questionSplit._nSubtasks);
      selQuestion = _dims._nQuestions - 1;
      break;
    }

    const taNumber inWorkerRunLen = selRunLen - ((iWorker == 0) ? SRDoubleNumber(0.0) : pGrandTotals[iWorker-1]);
    const TPqaId iFirst = ((iWorker == 0) ? 0 : questionSplit._pBounds[iWorker - 1]);
    const TPqaId iLimit = questionSplit._pBounds[iWorker];
    selQuestion = std::upper_bound(pRunLength + iFirst, pRunLength + iLimit, inWorkerRunLen) - pRunLength;
    if (selQuestion >= iLimit) {
      assert(selQuestion == iLimit);
      CELOG(Warning) << SR_FILE_LINE "Hopefully due to a rounding error, within-worker run length binary search hasn't"
        " found a strictly greater value, while the binary search over grand totals pointed to this worker's piece."
        " Random selection: " << selRunLen.ToAmount() << ", in-worker run length " << inWorkerRunLen.ToAmount()
        << ", worker index " << iWorker << ", grand total " << pGrandTotals[iWorker].ToAmount() << ", worker max run"
        " length " << pRunLength[iLimit-1].ToAmount();
      selQuestion = iLimit - 1;
    }
  } WHILE_FALSE;

  // If the selected question is in a gap or already answered, try to select the neighboring questions
  if (_questionGaps.IsGap(selQuestion) || SRBitHelper::Test(pQuiz->GetQAsked(), selQuestion)) {
    selQuestion = FindNearestQuestion(selQuestion, pQuiz->GetQAsked());
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

template<typename taNumber> TPqaId CpuEngine<taNumber>::ListTopTargetsSpec(PqaError& err, BaseQuiz *pBaseQuiz,
  const TPqaId maxCount, RatedTarget *pDest) 
{
  CEQuiz<taNumber> *pQuiz = static_cast<CEQuiz<taNumber>*>(pBaseQuiz);
  CEListTopTargetsAlgorithm<taNumber> ltta(err, *this, *pQuiz, maxCount, pDest);
  
  //TODO: experiment to determine operation weights (comparison vs memory operations).
  const uint64_t nTargPerThread = SRMath::PosDivideRoundUp<uint64_t>(ltta._nTargets, ltta._nWorkers);
  // This estimate is for algorithm that radix-sorts pieces, then uses a head heap to merge the pieces. However, there
  //   is also an much less cache friendly option to apply parallel radix sort to the whole array.
  const uint64_t nRadixSortOps = 9 * std::max<uint64_t>(nTargPerThread, ltta._cnRadixSortBuckets)
    + uint64_t(maxCount) * std::max(SRMath::CeilLog2(ltta._nWorkers), 1ui8);
  const uint64_t nHeapifyOps = 3 * nTargPerThread + uint64_t(maxCount) * SRMath::CeilLog2(ltta._nTargets);
  
  // Currently holds if maxCount > 6 * a / log2(a), where a=nTargets/nWorkers and a>=nRadixSortBuckets
  if (nRadixSortOps < nHeapifyOps) {
    return ltta.RunRadixSortBased();
    //CELOG(Warning) << "For " << ltta._nWorkers << " workers requested to list " << maxCount << " targets out of "
    //  << ltta._nTargets << ", which is a large enough part to prefer radix sort (" << nRadixSortOps << " Ops) over"
    //  " heapify (" << nHeapifyOps << " Ops) approach.";
  } else {
    return ltta.RunHeapifyBased();
  }
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RecordQuizTargetSpec(BaseQuiz *pBaseQuiz,
  const TPqaId iTarget, const TPqaAmount amount)
{
  CEQuiz<taNumber> *pQuiz = static_cast<CEQuiz<taNumber>*>(pBaseQuiz);
  const std::vector<AnsweredQuestion>& answers = pQuiz->GetAnswers();
  const CETrainTaskNumSpec<taNumber> numSpec(amount);
  CETrainOperation<taNumber> trainOp(*this, iTarget, numSpec);
  {
    SRRWLock<true> rwl(_rws);
    TPqaId i = 0;
    const TPqaId iEn = TPqaId(answers.size()) - 1;
    for (; i < iEn; i += 2) {
      const AnsweredQuestion& aqFirst = answers[i];
      const AnsweredQuestion& aqSecond = answers[i + 1];
      trainOp.Perform2(aqFirst, aqSecond);
    }
    assert(TPqaId(answers.size()) - 1 <= i && i <= TPqaId(answers.size()));
    if (i == iEn) {
      trainOp.Perform1(answers[i]);
    }
    _vB[iTarget] += amount;
  }

  return PqaError();
}

template<typename taNumber> PqaError CpuEngine<taNumber>::AddQsTsSpec(const TPqaId nQuestions, AddQuestionParam *pAqps,
  const TPqaId nTargets, AddTargetParam *pAtps)
{
  try {
    const TPqaId nQReuse = std::min(nQuestions, _questionGaps.GetNGaps());
    const TPqaId nQNew = nQuestions - nQReuse;
    const TPqaId nQOld = _dims._nQuestions;
    const TPqaId totQ = nQOld + nQNew;
    SRBitArray reusedQs(totQ, false);
    for (TPqaId i = 0; i < nQReuse; i++) {
      const TPqaId curQ = _questionGaps.Acquire();
      _pimQuestions.RenewComp(curQ);
      assert(curQ < _dims._nQuestions);
      pAqps[i]._iQuestion = curQ;
      reusedQs.SetOne(curQ);
    }

    const TPqaId nTReuse = std::min(nTargets, _targetGaps.GetNGaps());
    const TPqaId nTNew = nTargets - nTReuse;
    const TPqaId nTOld = _dims._nTargets;
    const TPqaId totT = nTOld + nTNew;
    for (TPqaId i = 0; i < nTReuse; i++) {
      const TPqaId curT = _targetGaps.Acquire();
      _pimTargets.RenewComp(curT);
      assert(curT < _dims._nTargets);
      pAtps[i]._iTarget = curT;
    }

    if (nQNew > 0) {
      _sA.resize(totQ);
      _mD.resize(totQ);
      for (TPqaId i = 0; i < nQNew; i++) {
        const TPqaId curQ = nQOld + i;
        pAqps[nQReuse + i]._iQuestion = curQ;
        const taNumber initSqr = taNumber(pAqps[nQReuse + i]._initialAmount).Sqr();
        _sA[curQ].resize(_dims._nAnswers);
        for (TPqaId k = 0; k < _dims._nAnswers; k++) {
          _sA[curQ][k].Resize<false>(totT);
          _sA[curQ][k].FillAll<false>(initSqr);
        }
        const taNumber initMD = initSqr * _dims._nAnswers;
        _mD[curQ].Resize<false>(totT);
        _mD[curQ].FillAll<false>(initMD);
      }
    }
    if (nTNew > 0) {
      for (TPqaId i = 0; i < nQOld; i++) {
        for (TPqaId k = 0; k < _dims._nAnswers; k++) {
          _sA[i][k].Resize<false>(totT);
          for (TPqaId j = 0; j < nTNew; j++) {
            const taNumber initSqr = taNumber(pAtps[nQReuse + j]._initialAmount).Sqr();
            _sA[i][k][j + nTOld] = initSqr; //TODO: vectorize and stream without caching
          }
        }
        _mD[i].Resize<false>(totT);
        for (TPqaId j = 0; j < nTNew; j++) {
          const taNumber initMD = taNumber(pAtps[nQReuse + j]._initialAmount).Sqr() * _dims._nAnswers;
          _mD[i][j + nTOld] = initMD; //TODO: vectorize and stream without caching
        }
      }
      _vB.Resize<false>(totT);
      for (TPqaId j = 0; j < nTNew; j++) {
        const TPqaId parPos = nQReuse + j;
        const TPqaId curT = nTOld + j;
        pAtps[parPos]._iTarget = curT;
        const taNumber init1(pAtps[parPos]._initialAmount);
        _vB[curT] = init1; //TODO: vectorize and stream without caching
      }
    }

    // As all the memory allocations have succeeded, it is safe to update the dimensions
    _dims._nQuestions = totQ;
    _dims._nTargets = totT;

    _pimQuestions.GrowTo(totQ);
    _pimTargets.GrowTo(totT);

    _questionGaps.GrowTo(totQ);
    _targetGaps.GrowTo(totT);

    // Set the initial amounts for questions and targets acquired from gaps
    for (TPqaId i = 0; i < nQReuse; i++) {
      const TPqaId curQ = pAqps[i]._iQuestion;
      const taNumber initSqr = taNumber(pAqps[i]._initialAmount).Sqr();
      for (TPqaId k = 0; k < _dims._nAnswers; k++) {
        _sA[curQ][k].FillAll<false>(initSqr);
      }
      const taNumber initMD = initSqr * _dims._nAnswers;
      _mD[curQ].FillAll<false>(initMD);
    }
    for (TPqaId j = 0; j < nTReuse; j++) {
      const TPqaId curT = pAtps[j]._iTarget;
      const taNumber init1(pAtps[j]._initialAmount);
      const taNumber initSqr = taNumber(init1).Sqr();
      const taNumber initMD = initSqr * _dims._nAnswers;
      for (TPqaId i = 0; i < nQOld; i++) {
        if (reusedQs.GetOne(i)) {
          continue; // already initialized by question initialization
        }
        for (TPqaId k = 0; k < _dims._nAnswers; k++) {
          _sA[i][k][curT] = initSqr;
        }
        _mD[i][curT] = initMD;
      }
      _vB[curT] = init1;
    }
    return PqaError();
  } CATCH_TO_ERR_RETURN;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::CompactSpec(CompactionResult &cr) {
  cr._nQuestions = _dims._nQuestions - _questionGaps.GetNGaps();
  const TPqaId nTargetGaps = _targetGaps.GetNGaps();
  cr._nTargets = _dims._nTargets - nTargetGaps;
  cr._pOldQuestions = static_cast<TPqaId*>(SRUtils::ThrowingSimdAlloc(SRSimd::PaddedBytesFromItems<sizeof(TPqaId)>(
    cr._nQuestions)));
  cr._pOldTargets = static_cast<TPqaId*>(SRUtils::ThrowingSimdAlloc(SRSimd::PaddedBytesFromItems<sizeof(TPqaId)>(
    cr._nTargets)));
  TPqaId iFirst, iLast;
  for (iFirst = 0, iLast = _dims._nQuestions - 1; iFirst <= iLast; iFirst++) {
    if (!_questionGaps.IsGap(iFirst)) {
      cr._pOldQuestions[iFirst] = iFirst;
      continue;
    }
    while (_questionGaps.IsGap(iLast) && iLast > iFirst) {
      iLast--;
    }
    if (iFirst == iLast) {
      break;
    }
    cr._pOldQuestions[iFirst] = iLast;
    _sA[iFirst].swap(_sA[iLast]);
    _mD[iFirst].Swap(_mD[iLast]);
    iLast--;
  }
  assert(iFirst == cr._nQuestions);
  _questionGaps.Compact(cr._nQuestions);
  _dims._nQuestions = cr._nQuestions;
  _pimQuestions.OnCompact(cr._nQuestions, cr._pOldQuestions);

  struct Move {
    TPqaId _iSrc;
    TPqaId _iDest;
  };
  SRSmartMPP<Move> moves(_memPool, nTargetGaps);
  TPqaId iGap = 0;
  for (iFirst = 0, iLast = _dims._nTargets - 1; iFirst <= iLast; iFirst++) {
    if (!_targetGaps.IsGap(iFirst)) {
      cr._pOldTargets[iFirst] = iFirst;
      continue;
    }
    while (_targetGaps.IsGap(iLast) && iLast > iFirst) {
      iLast--;
    }
    if (iFirst == iLast) {
      break;
    }
    moves.Get()[iGap]._iDest = iFirst;
    iGap++;
    moves.Get()[nTargetGaps - iGap]._iSrc = iLast;
    iLast--;
  }
  assert(iFirst == cr._nTargets);
  assert(iGap == nTargetGaps);

  for (iGap = 0; iGap < nTargetGaps; iGap++) {
    const Move &cm = moves.Get()[iGap];
    cr._pOldTargets[cm._iDest] = cm._iSrc;
    _vB[cm._iDest] = _vB[cm._iSrc];
  }
  for (TPqaId iQuestion = 0; iQuestion < cr._nQuestions; iQuestion++) {
    for (iGap = 0; iGap < nTargetGaps; iGap++) {
      const Move &cm = moves.Get()[iGap];
      _mD[iQuestion][cm._iDest] = _mD[iQuestion][cm._iSrc];
    }
  }
  for (TPqaId iQuestion = 0; iQuestion < cr._nQuestions; iQuestion++) {
    for (TPqaId iAnswer = 0; iAnswer < _dims._nAnswers; iAnswer++) {
      for (iGap = 0; iGap < nTargetGaps; iGap++) {
        const Move &cm = moves.Get()[iGap];
        _sA[iQuestion][iAnswer][cm._iDest] = _sA[iQuestion][iAnswer][cm._iSrc];
      }
    }
  }

  _targetGaps.Compact(cr._nTargets);
  _dims._nTargets = cr._nTargets;
  _pimTargets.OnCompact(cr._nTargets, cr._pOldTargets);

  return PqaError();
}

template<typename taNumber> size_t CpuEngine<taNumber>::NumberSize() {
  return sizeof(taNumber);
}

template<typename taNumber> PqaError CpuEngine<taNumber>::SaveStatistics(KBFileInfo &kbfi) {
  TargetRowPersistence<taNumber> trp(kbfi._sf, _dims._nTargets);
  for (TPqaId i = 0; i < _dims._nQuestions; i++) {
    for (TPqaId k = 0; k < _dims._nAnswers; k++) {
      if (!trp.Write(_sA[i][k])) {
        return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRMessageBuilder(SR_FILE_LINE
          "Can't write the target dimension of _sA weights at [")(i)(", ")(k)("].").GetOwnedSRString());
      }
    }
  }

  for (TPqaId i = 0; i < _dims._nQuestions; i++) {
    if (!trp.Write(_mD[i])) {
      return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRMessageBuilder(SR_FILE_LINE
        "Can't write the target dimension of _mD weights at [")(i)("].").GetOwnedSRString());
    }
  }

  if (!trp.Write(_vB)) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't write the _vB weights."));
  }

  return PqaError();
}

template<typename taNumber> PqaError CpuEngine<taNumber>::DestroyQuiz(BaseQuiz *pQuiz) {
  // Report error if the object is not of type CEQuiz<taNumber>
  CEQuiz<taNumber> *pSpecQuiz = dynamic_cast<CEQuiz<taNumber>*>(pQuiz);
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

template<typename taNumber> PqaError CpuEngine<taNumber>::DestroyStatistics() {
  _sA.clear();
  _mD.clear();
  _vB.Clear();
  return PqaError();
}

template<typename taNumber> void CpuEngine<taNumber>::UpdateWithDimensions() {
  const size_t newStackSize = CalcWorkerStackSize(_dims);
  if (newStackSize > _tpWorkers.GetStackSize()) {
    _tpWorkers.ChangeStackSize(newStackSize);
  }
}

//// Instantiations
template class CpuEngine<SRDoubleNumber>;

} // namespace ProbQA
