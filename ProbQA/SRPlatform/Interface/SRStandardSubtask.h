// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRBaseSubtask.h"
#include "../SRPlatform/Interface/SRBasicTypes.h"

namespace SRPlat {

// Derived classes must be constructible from single parameter: the pointer to a task. SRPoolRunner constructs them this
//   way.
class SRPLATFORM_API SRStandardSubtask : public SRBaseSubtask {
protected:
  int64_t _iFirst;
  int64_t _iLimit;
  SRSubtaskCount _iWorker;

public:
  explicit SRStandardSubtask(SRBaseTask *pTask) : SRBaseSubtask(pTask) { }

  virtual void SetStandardParams(const SRSubtaskCount iWorker, const int64_t iFirst, const int64_t iLimit) {
    _iFirst = iFirst;
    _iLimit = iLimit;
    _iWorker = iWorker;
  }
};

} // namespace SRPlat
