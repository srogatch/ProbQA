// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEBaseTask.fwd.h"
#include "../PqaCore/BaseCpuEngine.fwd.h"

namespace ProbQA {

class CEBaseTask : public SRPlat::SRBaseTask {
  BaseCpuEngine *_pCe;

public: // variables
  explicit CEBaseTask(BaseCpuEngine &engine);
  virtual SRPlat::SRThreadPool& GetThreadPool() const override final;
  BaseCpuEngine& GetBaseEngine() const;
};

} // namespace ProbQA
