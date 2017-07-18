// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRException.h"

namespace SRPlat {

class SRBaseTask;

class SRPLATFORM_API SRBaseSubtask {
  SRBaseTask *_pTask;

public:
  explicit SRBaseSubtask(SRBaseTask *pTask) : _pTask(pTask) { }
  virtual ~SRBaseSubtask() { }

  SRBaseTask *GetTask() const { return _pTask; }

  virtual void Run() = 0;
};

} // namespace SRPlat
