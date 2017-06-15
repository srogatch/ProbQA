#pragma once

#include "../PqaCore/CESubtask.h"

namespace ProbQA {

template<typename taNumber> class CETrainSubtaskAdd : public CESubtask<taNumber> {
public: // constants
  static const Kind _cKind = Kind::TrainAdd;

public: // variables

public: // methods
  virtual Kind GetKind() override {
    return _cKind;
  }
};

} // namespace ProbQA