#pragma once

#include "../PqaCore/CESubtask.h"

namespace ProbQA {

template<typename taNumber> class CETrainSubtaskDistrib : public CESubtask<taNumber> {
public: // variables
  const AnsweredQuestion *_pFirst;
  const AnsweredQuestion *_pLim;

public: // methods
  virtual Kind GetKind() override {
    return Kind::TrainDistrib;
  }
};

} // namespace ProbQA
