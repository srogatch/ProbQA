// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseEngine.h"

namespace ProbQA {

class BaseCudaEngine : public BaseEngine {
protected:
  explicit BaseCudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi);
};

} // namespace ProbQA
