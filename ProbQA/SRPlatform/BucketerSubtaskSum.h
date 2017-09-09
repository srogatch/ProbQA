// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRStandardSubtask.h"
#include "../SRPlatform/BucketerTask.h"
#include "../SRPlatform/Interface/SRRealNumber.h"

namespace SRPlat {

template<typename taNumber> class BucketerSubtaskSum : public SRStandardSubtask {
public: // types
  typedef BucketerTask<taNumber> TTask;

private: // variables
  const SRBucketSummator<taNumber> *_pBs;

private: // methods
  SRNumPack<taNumber> __vectorcall SumColumn(const size_t iVect);

public: // methods
  virtual void Run() override final;
};

} // namespace SRPlat
