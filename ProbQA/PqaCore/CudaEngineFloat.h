// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseCudaEngine.h"
#include "../PqaCore/CudaArray.h"

namespace ProbQA {

class CudaEngineFloat : public BaseCudaEngine {
public: // types
  typedef float TNumber;

private: // variables
  //// N questions, K answers, M targets
  // space A: [iQuestion][iAnswer][iTarget] . Guarded by _rws
  CudaArray<TNumber, true> _sA;
  // matrix D: [iQuestion][iTarget] . Guarded by _rws
  CudaArray<TNumber, true> _mD;
  // vector B: [iTarget] . Guarded by _rws
  CudaArray<TNumber, true> _vB;

public: // methods
  explicit CudaEngineFloat(const EngineDefinition& engDef, KBFileInfo *pKbFi);
};

} // namespace ProbQA
