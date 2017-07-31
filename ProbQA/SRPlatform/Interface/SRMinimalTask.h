// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRBaseTask.h"

namespace SRPlat {

class SRPLATFORM_API SRMinimalTask : public SRBaseTask {
  SRThreadPool *_pTp;
public:
  explicit SRMinimalTask(SRThreadPool &pTp) : _pTp(&pTp) { }
  virtual SRThreadPool& GetThreadPool() const override final { return *_pTp; }
};

} // namespace SRPlat
