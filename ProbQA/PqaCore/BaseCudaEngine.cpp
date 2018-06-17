// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseCudaEngine.h"
#include "../PqaCore/Interface/CudaMain.h"

namespace ProbQA {

BaseCudaEngine::BaseCudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi) : BaseEngine(engDef, pKbFi),
  _iDevice(0)
{
  CudaMain::Initialize(_iDevice);
}

} // namespace ProbQA
