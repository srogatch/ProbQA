#pragma once

#include "../PqaCore/Interface/IPqaEngine.h"
#include "../PqaCore/Interface/PqaErrors.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/Interface/PqaCore.h"

namespace ProbQA {

class PQACORE_API IPqaEngineFactory {
public:
  virtual IPqaEngine* CreateCpuEngine(PqaError& err, PrecisionDefinition precDef) = 0;
  virtual IPqaEngine* CreateCudaEngine(PqaError& err, PrecisionDefinition precDef) = 0;
};

extern "C" PQACORE_API IPqaEngineFactory& GetPqaEngineFactory();

} // namespace ProbQA
