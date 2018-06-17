// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CudaEngineFloat.h"
#include "../PqaCore/PqaException.h"

using namespace SRPlat;

namespace ProbQA {

CudaEngineFloat::CudaEngineFloat(const EngineDefinition& engDef, KBFileInfo *pKbFi) : BaseCudaEngine(engDef, pKbFi),
  _sA(size_t(engDef._dims._nQuestions) * engDef._dims._nAnswers * engDef._dims._nTargets),
  _mD(size_t(engDef._dims._nQuestions) * engDef._dims._nTargets),
  _vB(engDef._dims._nTargets)
{
  const TPqaId nQuestions = _dims._nQuestions;
  const TPqaId nAnswers = _dims._nAnswers;
  const TPqaId nTargets = _dims._nTargets;

  const TNumber init1 = TNumber(engDef._initAmount);
  const TNumber initSqr = TNumber(init1 * init1);
  const TNumber initMD = TNumber(initSqr * nAnswers);

  if (pKbFi == nullptr) { // init

  } else { // load
    const size_t nSAItems = int64_t(engDef._dims._nQuestions) * engDef._dims._nAnswers * engDef._dims._nTargets;
    if (std::fread(_sA.Get(), sizeof(TNumber), nSAItems, pKbFi->_sf.Get()) != nSAItems) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(pKbFi->_filePath), SRString::MakeUnowned(
        SR_FILE_LINE " Can't read cube A from file.")).ThrowMoving();
    }
    const size_t nMDItems = int64_t(engDef._dims._nQuestions) * engDef._dims._nTargets;
  }

  AfterStatisticsInit(pKbFi);
}

} // namespace ProbQA
