#pragma once

#include "../PqaCore/CESubtask.h"

namespace ProbQA {

template<typename taNumber> class CETrainSubtaskDistrib : public CESubtask<taNumber> {
public: // constants
  static const Kind _cKind = Kind::TrainDistrib;

public: // variables
  const AnsweredQuestion *_pFirst;
  const AnsweredQuestion *_pLim;

public: // methods
  virtual Kind GetKind() override {
    return _cKind;
  }
};

} // namespace ProbQA
