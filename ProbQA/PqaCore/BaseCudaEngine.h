// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseEngine.h"
#include "../PqaCore/CudaStreamPool.h"
#include "../PqaCore/CudaEngineGpu.cuh"
#include "../PqaCore/CudaArray.h"
#include "../PqaCore/CudaMemPool.h"

namespace ProbQA {

class BaseCudaEngine : public BaseEngine {
private: // variables
  KernelLaunchContext _klc;
  CudaArray<uint8_t> _gaps; // question, then target gaps
  CudaMemPool _cuMp;

protected: // variables
  const int _iDevice; // for now, use only one device
  CudaStreamPool _cspNb; // non-blocking CUDA stream pool

protected: // methods
  void BceUpdateWithDimensions(cudaStream_t stream);
  explicit BaseCudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi);
  PqaError ShutdownWorkers() override final { return PqaError(); };

public: // Internal interface methods
  int GetDevice() const { return _iDevice; }
  CudaStreamPool& GetCspNb() { return _cspNb; }
  uint8_t* DevQuestionGaps();
  uint8_t* DevTargetGaps();

public: // methods
  const KernelLaunchContext& GetKlc() const { return _klc; }
  CudaMemPool& GetCuMp() { return _cuMp; }
};

} // namespace ProbQA
