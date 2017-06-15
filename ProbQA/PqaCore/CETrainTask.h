#pragma once

#include "../PqaCore/CETask.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CETrainTask : public CETask<taNumber> {
public: // variables
  TPqaId *_prev;
  const TPqaAmount _amount;
  std::atomic<TPqaId> *_last;
  std::atomic<TPqaId> _iPrev;

public: // methods
  explicit CETrainTask(CpuEngine<taNumber> *pCe, const TPqaAmount amount) : CETask(pCe), _iPrev(0), _amount(amount)
  { }
  CETrainTask(const CETrainTask&) = delete;
  CETrainTask& operator=(const CETrainTask&) = delete;
  CETrainTask(CETrainTask&&) = delete;
  CETrainTask& operator=(CETrainTask&&) = delete;
};

} // namespace ProbQA