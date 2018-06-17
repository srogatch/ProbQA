// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CudaEngineFloat.h"

namespace ProbQA {

CudaEngineFloat::CudaEngineFloat(const EngineDefinition& engDef, KBFileInfo *pKbFi) : BaseCudaEngine(engDef, pKbFi),
  _sA(engDef._dims._nQuestions * engDef._dims._nAnswers * engDef._dims._nTargets),
  _mD(engDef._dims._nQuestions * engDef._dims._nTargets),
  _vB(engDef._dims._nTargets)
{
}

} // namespace ProbQA
