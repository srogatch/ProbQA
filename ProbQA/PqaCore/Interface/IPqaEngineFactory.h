// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/IPqaEngine.h"
#include "../PqaCore/Interface/PqaErrors.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/Interface/PqaCore.h"

namespace ProbQA {

class PQACORE_API IPqaEngineFactory {
public:
  // Usual computing on a CPU.
  virtual IPqaEngine* CreateCpuEngine(PqaError& err, const EngineDefinition& engDef) = 0;
  virtual IPqaEngine* LoadCpuEngine(PqaError& err, const char* const filePath,
    size_t memPoolMaxBytes = EngineDefinition::_cDefaultMemPoolMaxBytes) = 0;

  virtual PqaError SetCudaDevice(int iDevice, const bool bFirstInProcess) = 0;
  // Computing on a graphics card with CUDA technology.
  virtual IPqaEngine* CreateCudaEngine(PqaError& err, const EngineDefinition& engDef) = 0;
  virtual IPqaEngine* LoadCudaEngine(PqaError& err, const char* const filePath,
    size_t memPoolMaxBytes = EngineDefinition::_cDefaultMemPoolMaxBytes) = 0;

  // Grid computing over a network
  virtual IPqaEngine* CreateGridEngine(PqaError& err, const EngineDefinition& engDef) = 0;
};

PQACORE_API IPqaEngineFactory& PqaGetEngineFactory();

} // namespace ProbQA
