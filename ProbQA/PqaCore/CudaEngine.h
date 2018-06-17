// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseCudaEngine.h"
#include "../PqaCore/CudaArray.h"

namespace ProbQA {

template<typename taNumber> class CudaEngine : public BaseCudaEngine {
private: // variables
  //// N questions, K answers, M targets
  // space A: [iQuestion][iAnswer][iTarget] . Guarded by _rws
  CudaArray<taNumber, true> _sA;
  // matrix D: [iQuestion][iTarget] . Guarded by _rws
  CudaArray<taNumber, true> _mD;
  // vector B: [iTarget] . Guarded by _rws
  CudaArray<taNumber, true> _vB;

public: // methods
  explicit CudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi);
};

} // namespace ProbQA
