// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseCpuEngine.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

SRThreadCount BaseCpuEngine::CalcCompThreads() {
  //TODO: experiment whether nCores-1 threads give better performance than nCores threads. In the former case the
  //  master thread may keep spinning till worker threads are all done.
  return std::max(std::thread::hardware_concurrency()-1, 1u);
}

SRThreadCount BaseCpuEngine::CalcMemOpThreads() {
  // This is a trivial heuristic based on the observation that on Ryzen 1800X with 2 DDR4 modules in a single memory
  //   channel, the maximum copy speed is achieved for 5 threads.
  return std::max(1ui32, std::min(CalcCompThreads(), 5ui32));
}

BaseCpuEngine::BaseCpuEngine(const EngineDefinition& engDef)
  : _dims(engDef._dims), _maintSwitch(MaintenanceSwitch::Mode::Regular),
  _pLogger(SRDefaultLogger::Get()), _memPool(1 + (engDef._memPoolMaxBytes >> SRSimd::_cLogNBytes)),
  _tpWorkers(CalcCompThreads()), _nMemOpThreads(CalcMemOpThreads())
{
}

TPqaId BaseCpuEngine::FindNearestQuestion(const TPqaId iMiddle, const CEBaseQuiz &quiz) {
  constexpr uint8_t dInf = 200;
  const TPqaId iPack64 = iMiddle >> 6;
  const uint8_t iWithin = iMiddle & 63;
  const uint64_t available = ~(_questionGaps.GetPacked<uint64_t>(iPack64)
    | SRBitHelper::GetPacked<uint64_t>(quiz.GetQAsked(), iPack64));
  if (available != 0) {
    const uint64_t baseMask = (1ui64 << iWithin) - 1;
    const uint64_t higher = SRMath::AndNot(baseMask, available); // includes the bit itself
    const uint64_t lower = baseMask & available;
    unsigned long iSetBit;
    const uint32_t dHigher = (_BitScanForward64(&iSetBit, higher) ? (iSetBit-iWithin) : dInf);
    const uint32_t dLower = (_BitScanReverse64(&iSetBit, lower) ? (iWithin-iSetBit) : dInf);
    if (dHigher < dLower) {
      return iMiddle + dHigher;
    } else {
      return iMiddle - dLower;
    }
  }
  const TPqaId limPack64 = (_dims._nQuestions + 63) >> 6;
  TPqaId i = 1;
  while ((iPack64 >= i) && (iPack64 + i < limPack64)) {
    const uint64_t availLeft = ~(_questionGaps.GetPacked<uint64_t>(iPack64-i)
      | SRBitHelper::GetPacked<uint64_t>(quiz.GetQAsked(), iPack64-i));
    const uint64_t availRight = ~(_questionGaps.GetPacked<uint64_t>(iPack64 + i)
      | SRBitHelper::GetPacked<uint64_t>(quiz.GetQAsked(), iPack64 + i));
    if ((availLeft | availRight) == 0) {
      i++;
      continue;
    }
    unsigned long iSetBit;
    const uint32_t dHigher = (_BitScanForward64(&iSetBit, availRight) ?  (iSetBit + 64 - iWithin) : dInf);
    const uint32_t dLower = (_BitScanReverse64(&iSetBit, availLeft) ? (iWithin + 64 - iSetBit) : dInf);
    if (dHigher < dLower) {
      return iMiddle + dHigher + ((i - 1) << 6);
    } else {
      return iMiddle - dLower - ((i - 1) << 6);
    }
  }
  while (iPack64 >= i) {
    const uint64_t availLeft = ~(_questionGaps.GetPacked<uint64_t>(iPack64 - i)
      | SRBitHelper::GetPacked<uint64_t>(quiz.GetQAsked(), iPack64 - i));
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
      | SRBitHelper::GetPacked<uint64_t>(quiz.GetQAsked(), iPack64 + i));
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

} // namespace ProbQA
