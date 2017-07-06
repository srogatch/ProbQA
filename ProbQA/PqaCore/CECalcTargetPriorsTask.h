// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CETask.h"
#include "../PqaCore/CECalcTargetPriorsTaskNumSpec.h"

namespace ProbQA {

template<typename taNumber> class CECalcTargetPriorsTask : public CETask<taNumber> {
public:
  taNumber *_pDest;
  CECalcTargetPriorsTaskNumSpec<taNumber> _numSpec;

public:
  explicit CECalcTargetPriorsTask(CpuEngine<taNumber> *pEngine) : CETask(pEngine) { }
};

} // namespace ProbQA
