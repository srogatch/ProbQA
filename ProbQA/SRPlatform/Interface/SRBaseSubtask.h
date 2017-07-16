// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRException.h"

namespace SRPlat {

class SRBaseTask;

class SRBaseSubtask {
  SRBaseTask *_pTask;
  // Filled in if an exception, thrown from Run() method and descendants, reaches thread pool.
  std::unique_ptr<SRException> _pEx;

public:
  explicit SRBaseSubtask(SRBaseTask *pTask) : _pTask(pTask) { }
  virtual ~SRBaseSubtask() { }

  SRBaseTask *GetTask() const { return _pTask; }
  void SetException(SRException &&ex);

  virtual void Run() = 0;
};

} // namespace SRPlat
