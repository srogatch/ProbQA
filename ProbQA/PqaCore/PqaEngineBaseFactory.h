// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/IPqaEngineFactory.h"
#include "../PqaCore/Interface/PqaErrors.h"
 
namespace ProbQA {

class PqaEngineBaseFactory : public IPqaEngineFactory {
public:
  static const TPqaId _cMinAnswers = 2;
  static const TPqaId _cMinQuestions = 1;
  static const TPqaId _cMinTargets = 2;

public:
  IPqaEngine* CreateCpuEngine(PqaError& err, const EngineDefinition& engDef) override final;
  IPqaEngine* LoadCpuEngine(PqaError& err, const char* const filePath) override final;
  IPqaEngine* CreateCudaEngine(PqaError& err, const EngineDefinition& engDef) override final;
  IPqaEngine* CreateGridEngine(PqaError& err, const EngineDefinition& engDef) override final;
};

} // namespace ProbQA