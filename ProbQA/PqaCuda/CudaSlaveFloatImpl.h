// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCuda/CudaArray.h"

namespace PqaCuda {

class CudaSlaveFloatImpl {
private: // variables
  //// N questions, K answers, M targets
  // space A: [iQuestion][iAnswer][iTarget] . Guarded by _rws
  CudaArray _sA;
  // matrix D: [iQuestion][iTarget] . Guarded by _rws
  CudaArray _mD;
  // vector B: [iTarget] . Guarded by _rws
  CudaArray _vB;
  
  //_sA(sizeof(TNumber) * engDef._dims._nQuestions * engDef._dims._nAnswers * engDef._dims._nTargets),
  //_mD(sizeof(TNumber) * engDef._dims._nQuestions * engDef._dims._nTargets),
  //_vB(sizeof(TNumber) * engDef._dims._nTargets)
};

} // namespace PqaCuda
