// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRBaseSubtask.h"
#include "../SRPlatform/Interface/SRBaseTask.h"

namespace SRPlat {

void SRBaseSubtask::GuardedRun() {
  _pTask->OnSubtaskComplete(this);
}

} // namespace SRPlat
