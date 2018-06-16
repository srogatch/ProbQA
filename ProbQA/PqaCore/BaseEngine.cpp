// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseEngine.h"
#include "../PqaCore/ErrorHelper.h"
#include "../PqaCore/PqaException.h"

using namespace SRPlat;

namespace ProbQA {

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

void BaseEngine::LoadKBTail(KBFileInfo *pKbFi) {
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

  PqaError err;
  if (saveFilePath != nullptr) do {
    SRSmartFile sf(std::fopen(saveFilePath, "wb"));
    if (sf.Get() == nullptr) {
      err = PqaError(PqaErrorCode::CantOpenFile, new CantOpenFileErrorParams(saveFilePath), SRString::MakeUnowned(
        SR_FILE_LINE "Can't open the file to write KB to."));
      break;
    }
    KBFileInfo kbfi(sf, saveFilePath);
    err = LockedSaveKB(kbfi, false);
  } WHILE_FALSE;

  //// Release quizzes
  for (size_t i = 0; i < _quizzes.size(); i++) {
    if (_quizGaps.IsGap(i)) {
      continue;
    }
    SRCheckingRelease(_memPool, _quizzes[i]);
  }
  _quizzes.clear();
  _quizGaps.Compact(0);
  
  //TODO: check the order - perhaps some releases should happen while the workers are still operational

  //// Shutdown worker threads
  _tpWorkers.RequestShutdown();

  //// Release KB
  _sA.clear();
  _mD.clear();
  _vB.Clear();
  _questionGaps.Compact(0);
  _targetGaps.Compact(0);
  _dims._nAnswers = _dims._nQuestions = _dims._nTargets = 0;

  //// Release memory pool
  _memPool.FreeAllChunks();

  return err;
}


PqaError BaseEngine::LockedSaveKB(KBFileInfo &kbfi, const bool bDoubleBuffer)
{
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

  return PqaError();
}

} // namespace ProbQA
