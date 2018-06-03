// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/Interface/PqaCInterop.h"
#include "../PqaCore/Interface/IPqaEngineFactory.h"

using namespace ProbQA;

PQACORE_API void* CiPqaGetEngineFactory() {
  return static_cast<void*>(&(PqaGetEngineFactory()));
}

