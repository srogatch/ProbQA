#pragma once

#include "../PqaCore/CESubtask.h"

namespace ProbQA {

template<typename taNumber> class CECalcTargetPriorsSubtaskBase : public CESubtask<taNumber> {
public: // variables
  // first SIMD index
  TPqaId _iFirst;
  // limit SIMD index
  TPqaId _iLim;
};

template<typename taNumber, bool taCache> class CECalcTargetPriorsSubtask
  : public CECalcTargetPriorsSubtaskBase<taNumber>
{
public: // constants
  static const Kind _cKind = taCache ? Kind::CalcTargetPriorsCache : Kind::CalcTargetPriorsNocache;

public: // methods
  virtual Kind GetKind() override {
    return _cKind;
  }
};

} // namespace ProbQA
