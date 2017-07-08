// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"
#include "../SRPlatform/Interface/SRAlignedAllocator.h"
#include "../SRPlatform/Interface/SRSpinSync.h"
#include "../SRPlatform/Interface/SRSmartHandle.h"
#include "../SRPlatform/Interface/SRLogStream.h"
#include "../SRPlatform/Interface/SRMemPool.h"
#include "../SRPlatform/Interface/SRBitArray.h"
#include "../SRPlatform/Interface/SRFastRandom.h"
#include "../SRPlatform/Interface/SRFastArray.h"

namespace SRPlat {
  constexpr size_t test1 = sizeof(SRSpinSync<1>);
  std::vector<SRFastArray<double, false>> test2;

  void test3() {
    test2.push_back(SRFastArray<double, false>(10));
  }
}
