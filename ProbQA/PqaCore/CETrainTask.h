#pragma once

#include "../PqaCore/CETrainTaskNumberSpec.h"
#include "../PqaCore/CETask.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CETrainTask : public CETask<taNumber> {
public: // variables
  std::atomic<TPqaId> *_last;
  TPqaId *_prev;
  const AnsweredQuestion* const _pAQs;
  std::atomic<TPqaId> _iPrev;
  TPqaId _iTarget;
  CETrainTaskNumberSpec<taNumber> _numSpec;
  
public: // methods
  explicit CETrainTask(CpuEngine<taNumber> *pCe, const TPqaId iTarget, const AnsweredQuestion* const pAQs)
    : CETask(pCe), _iPrev(0), _iTarget(iTarget), _pAQs(pAQs) { }
  CETrainTask(const CETrainTask&) = delete;
  CETrainTask& operator=(const CETrainTask&) = delete;
  CETrainTask(CETrainTask&&) = delete;
  CETrainTask& operator=(CETrainTask&&) = delete;
};

} // namespace ProbQA
