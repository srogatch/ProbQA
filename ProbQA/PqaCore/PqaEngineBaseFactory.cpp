#include "stdafx.h"
#include "PqaEngineBaseFactory.h"
#include "../PqaCore/Interface/PqaErrorParams.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/DoubleNumber.h"
#include "../PqaCore/ErrorHelper.h"

using namespace SRPlat;

namespace ProbQA {

IPqaEngine* PqaEngineBaseFactory::CreateCpuEngine(PqaError& err, const EngineDefinition& engDef) {
  try {
    std::unique_ptr<IPqaEngine> pEngine;
    switch (engDef._prec._type) {
    case TPqaPrecisionType::Double:
      pEngine.reset(new CpuEngine<DoubleNumber>(engDef));
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
