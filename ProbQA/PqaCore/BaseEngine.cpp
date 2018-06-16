// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseEngine.h"

using namespace SRPlat;

namespace ProbQA {

BaseEngine::BaseEngine(const EngineDefinition& engDef) : _dims(engDef._dims), _precDef(engDef._prec),
  _maintSwitch(MaintenanceSwitch::Mode::Regular), _pLogger(SRDefaultLogger::Get()),
  _memPool(1 + (engDef._memPoolMaxBytes >> SRSimd::_cLogNBytes))
{

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

} // namespace ProbQA
