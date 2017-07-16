// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRSpinSync.h"

namespace SRPlat {

class SRBaseSubtask;

class SRBaseTask {
public: // types
  typedef SRPlat::SRSpinSync<16> TSync;

private: // variables

public: // methods
  void OnSubtaskComplete(SRBaseSubtask *pSubtask);

};

} // namespace SRPlat
