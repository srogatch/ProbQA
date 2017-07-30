// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CETask.h"
#include "../PqaCore/DoubleNumber.h"
#include "../PqaCore/CpuEngine.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> SRThreadPool& CETask<taNumber>::GetThreadPool() const {
  return _pCe->GetWorkers();
}

template class CETask<DoubleNumber>;

} // namespace ProbQA
