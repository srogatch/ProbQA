// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CudaEngineFloat.h"

namespace ProbQA {

CudaEngineFloat::CudaEngineFloat(const EngineDefinition& engDef, KBFileInfo *pKbFi) : BaseCudaEngine(engDef, pKbFi)
{
}

} // namespace ProbQA
