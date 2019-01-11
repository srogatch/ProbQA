// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseEngine.h"
#include "../PqaCore/ErrorHelper.h"
#include "../PqaCore/PqaException.h"
#include "../PqaCore/BaseQuiz.h"

using namespace SRPlat;

namespace ProbQA {

#define BELOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, _pLogger.load(std::memory_order_acquire))

BaseEngine::BaseEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi) : _dims(engDef._dims),
  _precDef(engDef._prec), _maintSwitch(MaintenanceSwitch::Mode::Regular), _pLogger(SRDefaultLogger::Get()),
  _memPool(1 + (engDef._memPoolMaxBytes >> SRSimd::_cLogNBytes))
{
  if (pKbFi != nullptr) {
    uint64_t nQuestionsAsked;
    if (std::fread(&nQuestionsAsked, sizeof(nQuestionsAsked), 1, pKbFi->_sf.Get()) != 1) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(SR_FILE_LINE
        "Can't read the number of questions asked.")).ThrowMoving();
    }
    _nQuestionsAsked.store(nQuestionsAsked, std::memory_order_release);
  }
}

void BaseEngine::AfterStatisticsInit(KBFileInfo *pKbFi) {
  _questionGaps.GrowTo(_dims._nQuestions);
  _targetGaps.GrowTo(_dims._nTargets);

  if (pKbFi != nullptr) {
    if (!ReadGaps(_questionGaps, *pKbFi)) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(SR_FILE_LINE
        "Can't read the question gaps.")).ThrowMoving();
    }
    if (!ReadGaps(_targetGaps, *pKbFi)) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(SR_FILE_LINE
        "Can't read the target gaps.")).ThrowMoving();
    }

    if (!_pimQuestions.Load(pKbFi->_sf.Get())) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(SR_FILE_LINE
        "Can't read the question permanent-compact ID mapping.")).ThrowMoving();
    }
    if (!_pimTargets.Load(pKbFi->_sf.Get())) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(SR_FILE_LINE
        "Can't read the target permanent-compact ID mapping.")).ThrowMoving();
    }
    if (!_pimQuizzes.Load(pKbFi->_sf.Get())) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(SR_FILE_LINE
        "Can't read the quizzes permanent-compact ID mapping.")).ThrowMoving();
    }
  }
}

TPqaId BaseEngine::FindNearestQuestion(const TPqaId iMiddle, const __m256i *pQAsked) {
  constexpr uint8_t dInf = 200;
  const TPqaId iPack64 = iMiddle >> 6;
  const uint8_t iWithin = iMiddle & 63;
  const uint64_t available = ~(_questionGaps.GetPacked<uint64_t>(iPack64)
    | SRBitHelper::GetPacked<uint64_t>(pQAsked, iPack64));
  if (available != 0) {
    const uint64_t baseMask = (1ui64 << iWithin) - 1;
    const uint64_t higher = SRMath::AndNot(baseMask, available); // includes the bit itself
    const uint64_t lower = baseMask & available;
    unsigned long iSetBit;
    const uint32_t dHigher = (_BitScanForward64(&iSetBit, higher) ? (iSetBit - iWithin) : dInf);
    const uint32_t dLower = (_BitScanReverse64(&iSetBit, lower) ? (iWithin - iSetBit) : dInf);
    if (dHigher < dLower) {
      return iMiddle + dHigher;
    }
    else {
      return iMiddle - dLower;
    }
  }
  const TPqaId limPack64 = (_dims._nQuestions + 63) >> 6;
  TPqaId i = 1;
  while ((iPack64 >= i) && (iPack64 + i < limPack64)) {
    const uint64_t availLeft = ~(_questionGaps.GetPacked<uint64_t>(iPack64 - i)
      | SRBitHelper::GetPacked<uint64_t>(pQAsked, iPack64 - i));
    const uint64_t availRight = ~(_questionGaps.GetPacked<uint64_t>(iPack64 + i)
      | SRBitHelper::GetPacked<uint64_t>(pQAsked, iPack64 + i));
    if ((availLeft | availRight) == 0) {
      i++;
      continue;
    }
    unsigned long iSetBit;
    const uint32_t dHigher = (_BitScanForward64(&iSetBit, availRight) ? (iSetBit + 64 - iWithin) : dInf);
    const uint32_t dLower = (_BitScanReverse64(&iSetBit, availLeft) ? (iWithin + 64 - iSetBit) : dInf);
    if (dHigher < dLower) {
      return iMiddle + dHigher + ((i - 1) << 6);
    }
    else {
      return iMiddle - dLower - ((i - 1) << 6);
    }
  }
  while (iPack64 >= i) {
    const uint64_t availLeft = ~(_questionGaps.GetPacked<uint64_t>(iPack64 - i)
      | SRBitHelper::GetPacked<uint64_t>(pQAsked, iPack64 - i));
    unsigned long iSetBit;
    if (!_BitScanReverse64(&iSetBit, availLeft)) {
      i++;
      continue;
    }
    const uint32_t dLower = iWithin + 64 - iSetBit;
    return iMiddle - dLower - ((i - 1) << 6);
  }
  while (iPack64 + i < limPack64) {
    const uint64_t availRight = ~(_questionGaps.GetPacked<uint64_t>(iPack64 + i)
      | SRBitHelper::GetPacked<uint64_t>(pQAsked, iPack64 + i));
    unsigned long iSetBit;
    if (!_BitScanForward64(&iSetBit, availRight)) {
      i++;
      continue;
    }
    const uint32_t dHigher = (iSetBit + 64 - iWithin);
    return iMiddle + dHigher + ((i - 1) << 6);
  }
  return cInvalidPqaId;
}

bool BaseEngine::ReadGaps(GapTracker<TPqaId> &gt, KBFileInfo &kbfi) {
  FILE *fpin = kbfi._sf.Get();
  TPqaId nGaps;
  if (std::fread(&nGaps, sizeof(nGaps), 1, fpin) != 1) {
    return false;
  }
  SRSmartMPP<TPqaId> gaps(_memPool, nGaps);
  if (TPqaId(std::fread(gaps.Get(), sizeof(TPqaId), nGaps, fpin)) != nGaps) {
    return false;
  }
  for (TPqaId i = 0; i < nGaps; i++) {
    gt.Release(gaps.Get()[i]);
  }
  return true;
}

bool BaseEngine::WriteGaps(const GapTracker<TPqaId> &gt, KBFileInfo &kbfi) {
  FILE *fpout = kbfi._sf.Get();
  const TPqaId nGaps = gt.GetNGaps();
  if (std::fwrite(&nGaps, sizeof(nGaps), 1, fpout) != 1) {
    return false;
  }
  if (TPqaId(std::fwrite(gt.ListGaps(), sizeof(TPqaId), nGaps, fpout)) != nGaps) {
    return false;
  }
  return true;
}

bool BaseEngine::QuestionPermFromComp(const TPqaId count, TPqaId *pIds) {
  MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
  SRRWLock<false> rwl(_rws);
  for (TPqaId i = 0; i < count; i++) {
    pIds[i] = _pimQuestions.PermFromComp(pIds[i]);
  }
  return true;
}

bool BaseEngine::QuestionCompFromPerm(const TPqaId count, TPqaId *pIds) {
  MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
  SRRWLock<false> rwl(_rws);
  for (TPqaId i = 0; i < count; i++) {
    pIds[i] = _pimQuestions.CompFromPerm(pIds[i]);
  }
  return true;
}

bool BaseEngine::TargetPermFromComp(const TPqaId count, TPqaId *pIds) {
  MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
  SRRWLock<false> rwl(_rws);
  for (TPqaId i = 0; i < count; i++) {
    pIds[i] = _pimTargets.PermFromComp(pIds[i]);
  }
  return true;
}

bool BaseEngine::TargetCompFromPerm(const TPqaId count, TPqaId *pIds) {
  MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
  SRRWLock<false> rwl(_rws);
  for (TPqaId i = 0; i < count; i++) {
    pIds[i] = _pimTargets.CompFromPerm(pIds[i]);
  }
  return true;
}

bool BaseEngine::QuizPermFromComp(const TPqaId count, TPqaId *pIds) {
  MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
  SRLock<SRCriticalSection> csl(_csQuizReg);
  for (TPqaId i = 0; i < count; i++) {
    pIds[i] = _pimQuizzes.PermFromComp(pIds[i]);
  }
  return true;
}

bool BaseEngine::QuizCompFromPerm(const TPqaId count, TPqaId *pIds) {
  MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
  SRLock<SRCriticalSection> csl(_csQuizReg);
  for (TPqaId i = 0; i < count; i++) {
    pIds[i] = _pimQuizzes.CompFromPerm(pIds[i]);
  }
  return true;
}

EngineDimensions BaseEngine::CopyDims() const {
  MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
  if (msal.GetMode() == MaintenanceSwitch::Mode::Regular) {
    return _dims;
  }
  //// Maintenance mode: dimensions may be modified concurrently
  SRRWLock<false> rwl(_rws);
  return _dims;
}

uint64_t BaseEngine::GetTotalQuestionsAsked(PqaError& err) {
  err.Release();
  return _nQuestionsAsked.load(std::memory_order_relaxed);
}

PqaError BaseEngine::Train(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
  const TPqaAmount amount)
{
  try {
    if (nQuestions < 0) {
      return PqaError(PqaErrorCode::NegativeCount, new NegativeCountErrorParams(nQuestions), SRString::MakeUnowned(
        "|nQuestions| must be non-negative."));
    }
    if (amount <= 0) {
      return PqaError(PqaErrorCode::NonPositiveAmount, new NonPositiveAmountErrorParams(amount), SRString::MakeUnowned(
        SR_FILE_LINE "|amount| must be positive."));
    }
    return TrainSpec(nQuestions, pAQs, iTarget, amount);
  }
  CATCH_TO_ERR_RETURN;
}

PqaError BaseEngine::SetLogger(ISRLogger *pLogger) {
  if (pLogger == nullptr) {
    pLogger = SRDefaultLogger::Get();
  }
  _pLogger.store(pLogger, std::memory_order_release);
  return PqaError();
}

PqaError BaseEngine::Shutdown(const char* const saveFilePath) {
  AggregateErrorParams aep;
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

  if (saveFilePath != nullptr) do {
    SRSmartFile sf(std::fopen(saveFilePath, "wb"));
    if (sf.Get() == nullptr) {
      aep.Add(PqaError(PqaErrorCode::CantOpenFile, new CantOpenFileErrorParams(saveFilePath), SRString::MakeUnowned(
        SR_FILE_LINE "Can't open the file to write KB to.")));
      break;
    }
    KBFileInfo kbfi(sf, saveFilePath);
    aep.Add(LockedSaveKB(kbfi, false));
    if (!sf.HardFlush()) {
      aep.Add(PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(saveFilePath), SRString::MakeUnowned(SR_FILE_LINE
        "Failed in hard flushing the KB when shutting down. See ProbQA log for details.")));
    }
  } WHILE_FALSE;

  //// Release quizzes
  for (size_t i = 0; i < _quizzes.size(); i++) {
    if (_quizGaps.IsGap(i)) {
      continue;
    }
    if (!_pimQuizzes.RemoveComp(i)) {
      aep.Add(PqaError(PqaErrorCode::Internal, new InternalErrorParams(__FILE__, __LINE__),
        SRString::MakeUnowned("Failed to remove quiz compact ID.") ));
    }
    aep.Add(DestroyQuiz(_quizzes[i]));
  }
  _quizzes.clear();
  _quizGaps.Compact(0);
  if (!_pimQuizzes.OnCompact(0, nullptr)) {
    aep.Add(PqaError(PqaErrorCode::Internal, new InternalErrorParams(__FILE__, __LINE__),
      SRString::MakeUnowned("Failed to compact the quiz permanent-compact ID mapper.") ));
  }
  
  //TODO: check the order - perhaps some releases should happen while the workers are still operational
  //// Shutdown worker threads
  aep.Add(ShutdownWorkers());

  //// Release KB
  aep.Add(DestroyStatistics());

  _questionGaps.Compact(0);
  _targetGaps.Compact(0);
  _dims._nAnswers = _dims._nQuestions = _dims._nTargets = 0;

  //// Release memory pool
  _memPool.FreeAllChunks();

  return aep.ToError(SRString::MakeUnowned(SR_FILE_LINE "Error(s) occured during shutdown."));
}

PqaError BaseEngine::LockedSaveKB(KBFileInfo &kbfi, const bool bDoubleBuffer) {
  const size_t nTargets = SRCast::ToSizeT(_dims._nTargets);

  size_t bufSize;
  if (bDoubleBuffer) {
    bufSize = /* reserve */ SRSimd::_cNBytes + sizeof(_precDef) + sizeof(_dims)
      + sizeof(decltype(_nQuestionsAsked)::value_type)
      + NumberSize() * nTargets * (_dims._nQuestions * _dims._nAnswers + _dims._nQuestions + 1);
  }
  else {
    bufSize = _cFileBufSize;
  }
  if (std::setvbuf(kbfi._sf.Get(), nullptr, _IOFBF, bufSize) != 0) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRMessageBuilder(SR_FILE_LINE
      "Can't set file buffer size to ")(bufSize).GetOwnedSRString());
  }

  // Can be out of locks so long that the member variable is const
  if (std::fwrite(&_precDef, sizeof(_precDef), 1, kbfi._sf.Get()) != 1) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't write precision definition header."));
  }

  if (std::fwrite(&_dims, sizeof(_dims), 1, kbfi._sf.Get()) != 1) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't write engine dimensions header."));
  }

  const uint64_t nQuestionsAsked = _nQuestionsAsked.load(std::memory_order_acquire);
  if (std::fwrite(&nQuestionsAsked, sizeof(nQuestionsAsked), 1, kbfi._sf.Get()) != 1) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't write the number of questions asked."));
  }

  PqaError err = SaveStatistics(kbfi);
  if (!err.IsOk()) {
    return err;
  }

  if (!WriteGaps(_questionGaps, kbfi)) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't write the question gaps."));
  }
  if (!WriteGaps(_targetGaps, kbfi)) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't write the target gaps."));
  }

  if (!_pimQuestions.Save(kbfi._sf.Get())) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't write the question permanent-compact ID mappings."));
  }
  if (!_pimTargets.Save(kbfi._sf.Get())) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't write the target permanent-compact ID mappings."));
  }
  if (!_pimQuizzes.Save(kbfi._sf.Get(), true)) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(kbfi._filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't write the quiz permanent-compact ID mappings."));
  }

  return PqaError();
}

TPqaId BaseEngine::ResumeQuiz(PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs) {
  if (nAnswered < 0) {
    err = PqaError(PqaErrorCode::NegativeCount, new NegativeCountErrorParams(nAnswered), SRString::MakeUnowned(
      "|nAnswered| must be non-negative."));
    return cInvalidPqaId;
  }
  if (nAnswered == 0) {
    return StartQuiz(err);
  }
  return ResumeQuizSpec(err, nAnswered, pAQs);
}

BaseQuiz* BaseEngine::UseQuiz(PqaError& err, const TPqaId iQuiz) {
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

TPqaId BaseEngine::NextQuestion(PqaError& err, const TPqaId iQuiz) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform regular-only"
      " mode operation (compute next question) because current mode is not regular (but maintenance/shutdown?)."));
    return cInvalidPqaId;
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  BaseQuiz *pQuiz = UseQuiz(err, iQuiz);
  if (pQuiz == nullptr) {
    assert(!err.IsOk());
    return cInvalidPqaId;
  }

  return NextQuestionSpec(err, pQuiz);
}

PqaError BaseEngine::RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    return PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform regular-only"
      " mode operation (record an answer) because current mode is not regular (but maintenance/shutdown?)."));
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  // Check that iAnswer is within the range
  if (iAnswer < 0 || iAnswer >= _dims._nAnswers) {
    return PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iAnswer, 0, _dims._nAnswers - 1),
      SRString::MakeUnowned("Answer index is not in the answer range."));
  }

  BaseQuiz *pQuiz;
  {
    PqaError err;
    pQuiz = UseQuiz(err, iQuiz);
    if (pQuiz == nullptr) {
      assert(!err.IsOk());
      return std::move(err);
    }
  }

  return pQuiz->RecordAnswer(iAnswer);
}

TPqaId BaseEngine::GetActiveQuestionId(PqaError &err, const TPqaId iQuiz) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform regular-only"
      " mode operation (get active question ID for a quiz) because current mode is not regular"
      " (but maintenance/shutdown?)."));
    return cInvalidPqaId;
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  BaseQuiz *pQuiz;
  {
    pQuiz = UseQuiz(err, iQuiz);
    if (pQuiz == nullptr) {
      assert(!err.IsOk());
      return cInvalidPqaId;
    }
  }

  return pQuiz->GetActiveQuestion();
}

PqaError BaseEngine::SetActiveQuestion(const TPqaId iQuiz, const TPqaId iQuestion) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    return PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform regular-only"
      " mode operation (get active question ID for a quiz) because current mode is not regular"
      " (but maintenance/shutdown?)."));
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  BaseQuiz *pQuiz;
  {
    PqaError err;
    pQuiz = UseQuiz(err, iQuiz);
    if (pQuiz == nullptr) {
      assert(!err.IsOk());
      return err;
    }
  }

  pQuiz->SetActiveQuestion(iQuestion);
  return PqaError();
}

TPqaId BaseEngine::ListTopTargets(PqaError& err, const TPqaId iQuiz, const TPqaId maxCount, RatedTarget *pDest) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    err = PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform regular-only"
      " mode operation (compute next question) because current mode is not regular (but maintenance/shutdown?)."));
    return cInvalidPqaId;
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  BaseQuiz *pQuiz = UseQuiz(err, iQuiz);
  if (pQuiz == nullptr) {
    assert(!err.IsOk());
    return cInvalidPqaId;
  }

  return ListTopTargetsSpec(err, pQuiz, maxCount, pDest);
}

PqaError BaseEngine::RecordQuizTarget(const TPqaId iQuiz, const TPqaId iTarget, const TPqaAmount amount) {
  if (amount <= 0) {
    return PqaError(PqaErrorCode::NonPositiveAmount, new NonPositiveAmountErrorParams(amount), SRString::MakeUnowned(
      SR_FILE_LINE "|amount| must be positive."));
  }

  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    return PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform regular-only"
      " mode operation (record quiz target) because current mode is not regular (but maintenance/shutdown?)."));
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  if (iTarget < 0 || iTarget >= _dims._nTargets) {
    const TPqaId nKB = _dims._nTargets;
    mssl.EarlyRelease();
    return PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iTarget, 0, nKB - 1),
      SRString::MakeUnowned(SR_FILE_LINE "Target index is not in KB range."));
  }

  if (_targetGaps.IsGap(iTarget)) {
    mssl.EarlyRelease();
    return PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iTarget), SRString::MakeUnowned(SR_FILE_LINE
      "Target index is not in KB (but rather at a gap)."));
  }

  BaseQuiz *pQuiz;
  {
    PqaError err;
    pQuiz = UseQuiz(err, iQuiz);
    if (pQuiz == nullptr) {
      assert(!err.IsOk());
      return std::move(err);
    }
  }

  return RecordQuizTargetSpec(pQuiz, iTarget, amount);
}

PqaError BaseEngine::ReleaseQuiz(const TPqaId iQuiz) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Regular;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    return PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform regular-only"
      " mode operation (release quiz) because current mode is not regular (but maintenance/shutdown?)."));
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);

  BaseQuiz *pQuiz;
  {
    SRLock<SRCriticalSection> csl(_csQuizReg);
    const TPqaId nQuizzes = _quizzes.size();
    if (iQuiz < 0 || iQuiz >= nQuizzes) {
      csl.EarlyRelease();
      // For nQuizzes == 0, this may return [0;-1] range: we can't otherwise return an empty range because we return
      //   the range with both bounds inclusive.
      return PqaError(PqaErrorCode::IndexOutOfRange, new IndexOutOfRangeErrorParams(iQuiz, 0, nQuizzes - 1),
        SRString::MakeUnowned(SR_FILE_LINE "Quiz index is not in quiz registry range."));
    }
    if (_quizGaps.IsGap(iQuiz)) {
      csl.EarlyRelease();
      return PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iQuiz), SRString::MakeUnowned(
        SR_FILE_LINE "Quiz index is not in the registry (but rather at a gap)."));
    }

    pQuiz = _quizzes[iQuiz];
    _quizzes[iQuiz] = nullptr; // avoid double-free problems

    _quizGaps.Release(iQuiz);
    if (!_pimQuizzes.RemoveComp(iQuiz)) {
      BELOG(Error) << SR_FILE_LINE << "Failed to remove quiz " << iQuiz << " from permanent-compact ID mapper.";
    }
  }

  return DestroyQuiz(pQuiz);
}


PqaError BaseEngine::SaveKB(const char* const filePath, const bool bDoubleBuffer) {
  SRSmartFile sf(std::fopen(filePath, "wb"));
  if (sf.Get() == nullptr) {
    return PqaError(PqaErrorCode::CantOpenFile, new CantOpenFileErrorParams(filePath), SRString::MakeUnowned(
      SR_FILE_LINE "Can't open the file to write KB to."));
  }

  KBFileInfo kbfi(sf, filePath);
  {
    MaintenanceSwitch::AgnosticLock msal(_maintSwitch);
    // Can't write engine dimensions before reader-writer lock, because maintenance switch doesn't prevent their change
    //   in maintenance mode.
    SRRWLock<false> rwl(_rws);

    PqaError err = LockedSaveKB(kbfi, bDoubleBuffer);
    if (!err.IsOk()) {
      return std::move(err);
    }
  }

  if (!sf.HardFlush()) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Failed in hard flushing the KB. See ProbQA log for details."));
  }
  // Close it explicitly here, to be able to handle and report an error
  if (!sf.EarlyClose()) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Failed in closing the file."));
  }

  return PqaError();
}


PqaError BaseEngine::StartMaintenance(const bool forceQuizzes) {
  try {
    constexpr auto cOrigMode = MaintenanceSwitch::Mode::Regular;
    constexpr auto cTargMode = MaintenanceSwitch::Mode::Maintenance;
    AggregateErrorParams aep;
    MaintenanceSwitch::SpecificLeaver<cTargMode> mssl;
    SRRWLock<true> rwl; // block every read and write until we are clear about quizzes
    SRLock<SRCriticalSection> csl;
    mssl = _maintSwitch.SwitchMode<cTargMode>([&]() {
      rwl.Init(_rws);
      csl.Init(_csQuizReg);
    });
    const TPqaId quizRange = _quizzes.size();
    const TPqaId nQuizzes = quizRange - _quizGaps.GetNGaps();
    assert(nQuizzes >= 0);
    if (nQuizzes != 0) {
      if (forceQuizzes) {
        for (TPqaId i = 0; i < quizRange; i++) {
          if (_quizGaps.IsGap(i)) {
            continue;
          }
          BaseQuiz *pQuiz = _quizzes[i];
          aep.Add(DestroyQuiz(pQuiz));
          _quizGaps.Release(i);
          if (!_pimQuizzes.RemoveComp(i)) {
            aep.Add(PqaError(PqaErrorCode::Internal, new InternalErrorParams(__FILE__, __LINE__),
              SRString::MakeUnowned("Failed to remove quiz compact ID.")));
          }
        }
        assert(TPqaId(_quizzes.size()) == _quizGaps.GetNGaps());
      }
      else {
        csl.EarlyRelease();
        mssl.EarlyRelease();
        _maintSwitch.SwitchMode<cOrigMode>();
        return PqaError(PqaErrorCode::QuizzesActive, new QuizzesActiveErrorParams(nQuizzes), SRString::MakeUnowned(
          SR_FILE_LINE "Can't switch to maintenance mode because there are active quizzes and forceQuizzes=false"));
      }
    }
    return aep.ToError(SRString::MakeUnowned(
      SR_FILE_LINE "Error(s) occurred during mode switch. Current mode can be any."));
  }
  CATCH_TO_ERR_RETURN;
}

PqaError BaseEngine::FinishMaintenance() {
  try {
    PqaError err;
    constexpr auto cTargMode = MaintenanceSwitch::Mode::Regular;
    _maintSwitch.SwitchMode<cTargMode>([&]() {
      try {
        // Adjust workers' stack size if more is needed for the new dimensions
        UpdateWithDimensions();
        // New dimensions will require new chunk sizes.
        //TODO: do this conditionally only if dimensions have changed.
        _memPool.FreeAllChunks();
      }
      CATCH_TO_ERR_SET(err);
    });
    return err;
  }
  CATCH_TO_ERR_RETURN;
}

PqaError BaseEngine::AddQsTs(const TPqaId nQuestions, AddQuestionParam *pAqps, const TPqaId nTargets,
  AddTargetParam *pAtps)
{
  constexpr auto msMode = MaintenanceSwitch::Mode::Maintenance;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    return PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform"
      " maintenance-only mode operation - add questions/targets - because current mode is not maintenance (but"
      " regular/shutdown?)."));
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);
  // Exclusive lock is needed because we are going to change the number of questions/targets in the KB, and also write
  //   the initial amounts.
  SRRWLock<true> rwl(_rws);

  return AddQsTsSpec(nQuestions, pAqps, nTargets, pAtps);
}

PqaError BaseEngine::RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds)
{
  constexpr auto msMode = MaintenanceSwitch::Mode::Maintenance;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    return PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform"
      " maintenance-only mode operation - remove questions - because current mode is not maintenance (but"
      " regular/shutdown?)."));
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);
  // Exclusive lock is needed because we are going to change the number of questions in the KB.
  SRRWLock<true> rwl(_rws);

  for (TPqaId i = 0; i < nQuestions; i++) {
    const TPqaId iQuestion = pQIds[i];
    if (iQuestion >= _dims._nQuestions || _questionGaps.IsGap(iQuestion)) {
      return PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iQuestion), SRString::MakeUnowned(SR_FILE_LINE
        "Question index is not in KB."));
    }
    _questionGaps.Release(iQuestion);
    _pimQuestions.RemoveComp(iQuestion);
  }
  return PqaError();
}

PqaError BaseEngine::RemoveTargets(const TPqaId nTargets, const TPqaId *pTIds) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Maintenance;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    return PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform"
      " maintenance-only mode operation - remove targets - because current mode is not maintenance (but"
      " regular/shutdown?)."));
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);
  // Exclusive lock is needed because we are going to change the number of targets in the KB.
  SRRWLock<true> rwl(_rws);

  for (TPqaId i = 0; i < nTargets; i++) {
    const TPqaId iTarget = pTIds[i];
    if (iTarget >= _dims._nTargets || _targetGaps.IsGap(iTarget)) {
      return PqaError(PqaErrorCode::AbsentId, new AbsentIdErrorParams(iTarget), SRString::MakeUnowned(SR_FILE_LINE
        "Target index is not in KB (but rather at a gap)."));
    }
    _targetGaps.Release(iTarget);
    _pimTargets.RemoveComp(iTarget);
  }
  return PqaError();
}

PqaError BaseEngine::Compact(CompactionResult &cr) {
  constexpr auto msMode = MaintenanceSwitch::Mode::Maintenance;
  if (!_maintSwitch.TryEnterSpecific<msMode>()) {
    return PqaError(PqaErrorCode::WrongMode, nullptr, SRString::MakeUnowned(SR_FILE_LINE "Can't perform"
      " maintenance-only mode operation - compact the KB - because current mode is not maintenance (but"
      " regular/shutdown?)."));
  }
  MaintenanceSwitch::SpecificLeaver<msMode> mssl(_maintSwitch);
  // Exclusive lock is needed because we are going to change the number of targets and questions in the KB.
  SRRWLock<true> rwl(_rws);
  return CompactSpec(cr);
}

TPqaId BaseEngine::AssignQuiz(BaseQuiz *pQuiz) {
  SRLock<SRCriticalSection> csl(_csQuizReg);
  const TPqaId quizId = _quizGaps.Acquire();
  if (quizId >= TPqaId(_quizzes.size())) {
    assert(quizId == TPqaId(_quizzes.size()));
    _quizzes.emplace_back(nullptr);
    _pimQuizzes.GrowTo(_quizzes.size());
  }
  else {
    _pimQuizzes.RenewComp(quizId);
  }
  _quizzes[SRCast::ToSizeT(quizId)] = pQuiz;
  return quizId;
}

void BaseEngine::UnassignQuiz(const TPqaId iQuiz) {
  SRLock<SRCriticalSection> csl(_csQuizReg);
  _quizzes[SRCast::ToSizeT(iQuiz)] = nullptr;
  _quizGaps.Release(iQuiz);
  _pimQuizzes.RemoveComp(iQuiz);
}

} // namespace ProbQA
