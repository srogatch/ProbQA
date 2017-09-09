// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRStandardSubtask.h"
#include "../SRPlatform/BucketerTask.h"

namespace SRPlat {

template<typename taNumber> class BucketerSubtaskSum : public SRStandardSubtask {
public: // types
  typedef BucketerTask<taNumber> TTask;

public: // methods
  virtual void Run() override final;
};

} // namespace SRPlat
