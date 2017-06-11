#pragma once

#include "../PqaCore/CETask.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CETrainTask : public CETask<taNumber> {
public: // variables
  std::unique_ptr<TPqaId[]> _prev;
  std::atomic<TPqaId> _iPrev;
  std::unique_ptr<std::atomic<TPqaId>[]> _last;

public: // methods
  explicit CETrainTask(CpuEngine<taNumber> *pCe) : CETask(pCe), _iPrev(0) { }
};

} // namespace ProbQA