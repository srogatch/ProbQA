// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace SRPlat {

class SRBaseSubtask;
class SRBaseTask;

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

  ~SubtaskCompleter();
};

} // namespace SRPlat