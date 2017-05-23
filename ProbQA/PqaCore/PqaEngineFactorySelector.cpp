#include "stdafx.h"
#include "PqaEngineBaseFactory.h"
#include "PqaCore.h"

namespace ProbQA {

PqaEngineBaseFactory gPqaEbf;

extern "C" PQACORE_API IPqaEngineFactory& GetPqaEngineFactory() {
  return gPqaEbf;
}

} // namespace ProbQA
