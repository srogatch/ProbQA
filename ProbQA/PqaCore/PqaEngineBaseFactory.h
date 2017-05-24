#pragma once

#include "../PqaCore/Interface/IPqaEngineFactory.h"
#include "../PqaCore/Interface/PqaErrors.h"
 
namespace ProbQA {

class PqaEngineBaseFactory : public IPqaEngineFactory {
public:
  IPqaEngine* CreateCpuEngine(PqaError& err, PrecisionDefinition precDef) override;
  IPqaEngine* CreateCudaEngine(PqaError& err, PrecisionDefinition precDef) override;
};

} // namespace ProbQA