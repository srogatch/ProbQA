// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseCudaEngine.h"

namespace ProbQA {

class CudaEngineFloat : public BaseCudaEngine {
public: // types
  typedef float TNumber;

private: // variables

public: // methods
  explicit CudaEngineFloat(const EngineDefinition& engDef, KBFileInfo *pKbFi);
};

} // namespace ProbQA
