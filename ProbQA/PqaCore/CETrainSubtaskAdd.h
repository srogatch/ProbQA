#pragma once

#include "../PqaCore/CESubtask.h"

namespace ProbQA {

template<typename taNumber> class CETrainSubtaskAdd : public CESubtask<taNumber> {
public: // variables


public: // methods
  virtual Kind GetKind() override {
    return Kind::TrainAdd;
  }
};

} // namespace ProbQA