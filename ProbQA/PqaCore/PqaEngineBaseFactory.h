// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

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