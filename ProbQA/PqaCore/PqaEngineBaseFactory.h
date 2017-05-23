#pragma once

#include "IPqaEngineFactory.h"

namespace ProbQA {

class PqaEngineBaseFactory : public IPqaEngineFactory {
public:
  IPqaEngine* CreateCpuEngine(PrecisionDefinition precDef) override;
  IPqaEngine* CreateCudaEngine(PrecisionDefinition precDef) override;
};

} // namespace ProbQA