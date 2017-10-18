// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "PqaEngineBaseFactory.h"
#include "../PqaCore/Interface/PqaErrorParams.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/ErrorHelper.h"

using namespace SRPlat;

namespace ProbQA {

IPqaEngine* PqaEngineBaseFactory::CreateCpuEngine(PqaError& err, const EngineDefinition& engDef) {
  if (engDef._dims._nAnswers < _cMinAnswers || engDef._dims._nQuestions < _cMinQuestions
    || engDef._dims._nTargets < _cMinTargets)
  {
    err = PqaError(PqaErrorCode::InsufficientEngineDimensions, new InsufficientEngineDimensionsErrorParams(
      engDef._dims._nAnswers, _cMinAnswers, engDef._dims._nQuestions, _cMinQuestions, engDef._dims._nTargets,
      _cMinTargets));
    return nullptr;
  }
  try {
    std::unique_ptr<IPqaEngine> pEngine;
    switch (engDef._prec._type) {
    case TPqaPrecisionType::Double:
      pEngine.reset(new CpuEngine<SRDoubleNumber>(engDef));
      break;
    default:
      //TODO: implement
      err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
        "ProbQA Engine on CPU for precision except double.")));
      return nullptr;
    }
    err.Release();
    return pEngine.release();
  }
  CATCH_TO_ERR_SET(err);
  return nullptr;
}

IPqaEngine* PqaEngineBaseFactory::CreateCudaEngine(PqaError& err, const EngineDefinition& engDef) {
  (void)engDef; //TODO: remove when implemented
  //TODO: implement
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "ProbQA Engine on CUDA.")));
  return nullptr;
}

IPqaEngine* PqaEngineBaseFactory::CreateGridEngine(PqaError& err, const EngineDefinition& engDef) {
  (void)engDef; //TODO: remove when implemented
  //TODO: implement
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "ProbQA Engine over a grid.")));
  return nullptr;
}

} // namespace ProbQA
