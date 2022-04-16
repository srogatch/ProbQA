// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/SRFastRandom.h"

namespace SRPlat {

namespace {
  thread_local SRFastRandom gTlRandom;
}

SRFastRandom& SRFastRandom::ThreadLocal() {
  return gTlRandom;
}

} // namespace SRPlat
