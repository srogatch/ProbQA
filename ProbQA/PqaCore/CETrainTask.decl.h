// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CETrainTask.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CETrainTaskNumSpec.h"
#include "../PqaCore/CETask.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CETrainTask : public CETask {
public: // variables
  std::atomic<TPqaId> *_last;
  TPqaId *_prev;
  const AnsweredQuestion* const _pAQs;
  std::atomic<TPqaId> _iPrev;
  TPqaId _iTarget;
  CETrainTaskNumSpec<taNumber> _numSpec;

public: // methods
  explicit CETrainTask(CpuEngine<taNumber> &ce, const SRPlat::SRThreadCount nWorkers,
    const TPqaId iTarget, const AnsweredQuestion* const pAQs);
  CETrainTask(const CETrainTask&) = delete;
  CETrainTask& operator=(const CETrainTask&) = delete;
  CETrainTask(CETrainTask&&) = delete;
  CETrainTask& operator=(CETrainTask&&) = delete;
};

} // namespace ProbQA
