#pragma once

#include "IPqaEngine.h"
#include "PqaCommon.h"
#include "PqaCore.h"

namespace ProbQA {

class PQACORE_API IPqaEngineFactory {
public:
  virtual IPqaEngine* CreateCpuEngine(PrecisionDefinition precDef) = 0;
  virtual IPqaEngine* CreateCudaEngine(PrecisionDefinition precDef) = 0;
};

extern "C" PQACORE_API IPqaEngineFactory& GetPqaEngineFactory();

} // namespace ProbQA
