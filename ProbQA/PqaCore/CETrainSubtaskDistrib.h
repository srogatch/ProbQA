// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

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
