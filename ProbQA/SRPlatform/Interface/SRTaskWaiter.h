// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRBaseTask.h"

namespace SRPlat {

class SRTaskWaiter {
  SRBaseTask *_pTask;
public:
  explicit SRTaskWaiter(SRBaseTask *pTask = nullptr) : _pTask(pTask) { }
  void SetTask(SRBaseTask *pTask = nullptr) { _pTask = pTask; }
  ~SRTaskWaiter() {
    if(_pTask != nullptr) {
      _pTask->WaitComplete();
    }
  }
};

} // namespace SRPlat
