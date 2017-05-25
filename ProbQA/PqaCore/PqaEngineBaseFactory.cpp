#include "stdafx.h"
#include "PqaEngineBaseFactory.h"
#include "../PqaCore/Interface/PqaErrorParams.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/DoubleNumber.h"

namespace ProbQA {

IPqaEngine* PqaEngineBaseFactory::CreateCpuEngine(PqaError& err, const EngineDefinition& engDef) {
  //TODO: implement
  switch (engDef._prec._type) {
  case TPqaPrecisionType::Double:
    return new CpuEngine<DoubleNumber>(engDef);
  default:
    err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams("ProbQA Engine on CPU for precision"
      " except double."));
    return nullptr;
  }
}

IPqaEngine* PqaEngineBaseFactory::CreateCudaEngine(PqaError& err, const EngineDefinition& engDef) {
  //TODO: implement
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams("ProbQA Engine on CUDA."));
  return nullptr;
}

IPqaEngine* CreateGridEngine(PqaError& err, const EngineDefinition& engDef) {
  //TODO: implement
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams("ProbQA Engine over a grid."));
  return nullptr;
}

} // namespace ProbQA
