// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "PqaEngineBaseFactory.h"
#include "../PqaCore/Interface/PqaCore.h"

namespace ProbQA {

PqaEngineBaseFactory gPqaEbf;

PQACORE_API IPqaEngineFactory& PqaGetEngineFactory() {
  return gPqaEbf;
}

} // namespace ProbQA
