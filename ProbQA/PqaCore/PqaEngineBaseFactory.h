#pragma once

#include "../PqaCore/Interface/IPqaEngineFactory.h"
#include "../PqaCore/Interface/PqaErrors.h"
 
namespace ProbQA {

class PqaEngineBaseFactory : public IPqaEngineFactory {
public:
  IPqaEngine* CreateCpuEngine(PqaError& err, const EngineDefinition& engDef) override;
  IPqaEngine* CreateCudaEngine(PqaError& err, const EngineDefinition& engDef) override;
  IPqaEngine* CreateGridEngine(PqaError& err, const EngineDefinition& engDef) override;
};

} // namespace ProbQA