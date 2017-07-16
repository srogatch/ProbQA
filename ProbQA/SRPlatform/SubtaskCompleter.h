// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRBaseSubtask.h"
#include "../SRPlatform/Interface/SRBaseTask.h"

namespace SRPlat {

class SubtaskCompleter {
  SRBaseSubtask *_pSubtask = nullptr;

public:
  SubtaskCompleter() {}
  SubtaskCompleter(const SubtaskCompleter&) = delete;
  SubtaskCompleter& operator=(const SubtaskCompleter&) = delete;
  SubtaskCompleter(SubtaskCompleter&&) = delete;
  SubtaskCompleter& operator=(SubtaskCompleter&&) = delete;

  void Set(SRBaseSubtask *pSubtask) { _pSubtask = pSubtask; }

  SRBaseSubtask* Get() const { return _pSubtask; }

  SRBaseSubtask* Detach() {
    SRBaseSubtask *answer = _pSubtask;
    _pSubtask = nullptr;
    return answer;
  }

  ~SubtaskCompleter() {
    if (_pSubtask != nullptr) {
      SRBaseTask *pTask = _pSubtask->GetTask();
      // Can't use subtask after the call below.
      pTask->OnSubtaskComplete(_pSubtask);
    }
  }
};

} // namespace SRPlat