// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/IPqaEngineFactory.h"
#include "../PqaCore/Interface/PqaErrors.h"
#include "../PqaCore/KBFileInfo.h"
 
namespace ProbQA {

class PqaEngineBaseFactory : public IPqaEngineFactory {
public: // constants
  static const TPqaId _cMinAnswers = 2;
  static const TPqaId _cMinQuestions = 1;
  static const TPqaId _cMinTargets = 2;

private: // methods
  IPqaEngine* MakeCpuEngine(PqaError& err, const EngineDefinition& engDef, KBFileInfo *pKbFi);

public: // methods
  IPqaEngine* CreateCpuEngine(PqaError& err, const EngineDefinition& engDef) override final;
  IPqaEngine* LoadCpuEngine(PqaError& err, const char* const filePath) override final;

  IPqaEngine* CreateCudaEngine(PqaError& err, const EngineDefinition& engDef) override final;
  IPqaEngine* CreateGridEngine(PqaError& err, const EngineDefinition& engDef) override final;
};

} // namespace ProbQA
