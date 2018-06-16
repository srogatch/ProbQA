// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseCpuEngine.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

SRThreadCount BaseCpuEngine::CalcMemOpThreads() {
  // This is a trivial heuristic based on the observation that on Ryzen 1800X with 2 DDR4 modules in a single memory
  //   channel, the maximum copy speed is achieved for 5 threads.
  return std::max(1ui32, std::min(std::thread::hardware_concurrency(), 5ui32));
}

BaseCpuEngine::BaseCpuEngine(const EngineDefinition& engDef, const size_t workerStackSize, KBFileInfo *pKbFi)
  : BaseEngine(engDef, pKbFi),
  _tpWorkers(std::thread::hardware_concurrency(), workerStackSize), _nMemOpThreads(CalcMemOpThreads()),
  _nLooseWorkers(std::max<SRThreadCount>(1, std::thread::hardware_concurrency()-1))
{
  _pimQuestions.GrowTo(_dims._nQuestions);
  _pimTargets.GrowTo(_dims._nTargets);
}

} // namespace ProbQA
