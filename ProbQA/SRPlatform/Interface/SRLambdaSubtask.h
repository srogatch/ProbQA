// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRBaseSubtask.h"

namespace SRPlat {

// taFunc must be a lambda (maybe also an object with operator() or std::function) that takes SRBaseSubtask reference
//   as the single parameter.
template<typename taFunc> class SRLambdaSubtask : public SRBaseSubtask {
  taFunc _f;

public:
  SRLambdaSubtask(SRBaseTask *pTask, taFunc &&f) : SRBaseSubtask(pTask), _f(std::forward<taFunc>(f)) { }
  SRLambdaSubtask(SRLambdaSubtask&) = delete;
  SRLambdaSubtask& operator=(const SRLambdaSubtask&) = delete;
  SRLambdaSubtask(SRLambdaSubtask&&) = delete;
  SRLambdaSubtask& operator=(SRLambdaSubtask&&) = delete;

  virtual void Run() override final { _f(*this); }
};

template <typename taFunc> inline SRLambdaSubtask<std::decay_t<taFunc>>
SRMakeLambdaSubtask(SRBaseTask *pTask, taFunc&& f) {
  return { pTask, std::forward<taFunc>(f) };
}

} // namespace SRPlat
