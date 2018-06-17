// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseEngine.h"
#include "../PqaCore/CudaStreamPool.h"
#include "../PqaCore/CudaEngineGpu.cuh"

namespace ProbQA {

class BaseCudaEngine : public BaseEngine {
private: // variables
  KernelLaunchContext _klc;

protected:
  const int _iDevice; // for now, use only one device
  CudaStreamPool _cspNb; // non-blocking CUDA stream pool

protected:
  explicit BaseCudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi);

public:
  const KernelLaunchContext& GetKlc() const { return _klc; }
};

} // namespace ProbQA
