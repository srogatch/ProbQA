#pragma once

#include "../PqaCore/CESubtask.h"

namespace ProbQA {

template<typename taNumber> class CECalcTargetPriorsSubtask : public CESubtask<taNumber> {
public: // constants
  static const Kind _cKind = Kind::CalcTargetPriors;

public: // variables
  TPqaId _iFirst;
  TPqaId _iLim;

public: // methods
  virtual Kind GetKind() override {
    return _cKind;
  }
};

} // namespace ProbQA