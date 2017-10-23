// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CETrainTask.decl.h"

namespace ProbQA {

template<typename taNumber> inline CETrainTask<taNumber>::CETrainTask(CpuEngine<taNumber> &ce,
  const SRPlat::SRThreadCount nWorkers, const TPqaId iTarget, const AnsweredQuestion* const pAQs,
  const TPqaAmount amount) : CETask(ce, nWorkers), _iPrev(0), _iTarget(iTarget), _pAQs(pAQs), _numSpec(amount)
{ }

} // namespace ProbQA
