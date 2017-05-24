#include "stdafx.h"
#include "PqaEngineBaseFactory.h"

namespace ProbQA {

IPqaEngine* PqaEngineBaseFactory::CreateCpuEngine(PqaError& err, PrecisionDefinition precDef) {
  //TODO: implement
}

IPqaEngine* PqaEngineBaseFactory::CreateCudaEngine(PqaError& err, PrecisionDefinition precDef) {
  //TODO: implement
}

} // namespace ProbQA
