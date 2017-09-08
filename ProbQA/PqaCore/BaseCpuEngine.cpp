// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseCpuEngine.h"

using namespace SRPlat;

namespace ProbQA {

SRThreadCount BaseCpuEngine::CalcCompThreads() {
  return std::thread::hardware_concurrency();
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

} // namespace ProbQA
