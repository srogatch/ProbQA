// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRVectMath.h"
#include "../SRPlatform/Interface/SRBucketSummatorSeq.h"
#include "../SRPlatform/Interface/SRPoolRunner.h"
#include "../SRPlatform/Interface/SRStandardSubtask.h"
#include "../SRPlatform/Interface/SRBucketSummatorPar.h"
#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRQueue.h"
#include "../SRPlatform/Interface/SRTaskWaiter.h"
#include "../SRPlatform/Interface/SRCpuInfo.h"
#include "../SRPlatform/Interface/SRMinimalTask.h"
#include "../SRPlatform/Interface/SRLambdaSubtask.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"
#include "../SRPlatform/Interface/SRAlignedAllocator.h"
#include "../SRPlatform/Interface/SRSpinSync.h"
#include "../SRPlatform/Interface/SRSmartHandle.h"
#include "../SRPlatform/Interface/SRLogStream.h"
#include "../SRPlatform/Interface/SRMemPool.h"
#include "../SRPlatform/Interface/SRBitArray.h"
#include "../SRPlatform/Interface/SRFastRandom.h"
#include "../SRPlatform/Interface/SRFastArray.h"
#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRThreadPool.h"
#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {
  constexpr size_t test1 = sizeof(SRSpinSync<1>);
  std::vector<SRFastArray<double, false>> test2;

  void test3() {
    test2.push_back(SRFastArray<double, false>(10));
    test2[0].Fill<false>(0, 10, 3.33);
    test2[0].Resize<false>(20);
    test2[0].FillAll<true>(7.77);
  }
}
