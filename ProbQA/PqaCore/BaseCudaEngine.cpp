// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseCudaEngine.h"
#include "../PqaCore/Interface/CudaMain.h"
#include "../PqaCore/CudaMacros.h"

namespace ProbQA {

BaseCudaEngine::BaseCudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi) : BaseEngine(engDef, pKbFi),
  _iDevice(0), _cspNb(false)
{
  CudaMain::Initialize(_iDevice);
  cudaDeviceProp cdp;
  CUDA_MUST(cudaGetDeviceProperties(&cdp, _iDevice));
  _klc._logBlockSize = 8;
  _klc._maxBlocks = int32_t((int64_t(cdp.multiProcessorCount) * cdp.maxThreadsPerMultiProcessor) >> _klc._logBlockSize);
}

} // namespace ProbQA
