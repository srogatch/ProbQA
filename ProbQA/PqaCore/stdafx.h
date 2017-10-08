// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>
#undef min
#undef max

// CPU-specific header files
#include <immintrin.h>
#include <intrin.h>

// STL
#pragma warning( push )
#pragma warning( disable : 4251 ) // needs to have dll-interface to be used by clients of class
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>
#pragma warning( pop )

// SRPlatform library includes
#include "../SRPlatform/Interface/ISRLogger.h"
#include "../SRPlatform/Interface/SRAccumulator.h"
#include "../SRPlatform/Interface/SRAlignedAllocator.h"
#include "../SRPlatform/Interface/SRBaseSubtask.h"
#include "../SRPlatform/Interface/SRBaseTask.h"
#include "../SRPlatform/Interface/SRBasicTypes.h"
#include "../SRPlatform/Interface/SRBitArray.h"
#include "../SRPlatform/Interface/SRBucketSummatorPar.h"
#include "../SRPlatform/Interface/SRBucketSummatorSeq.h"
#include "../SRPlatform/Interface/SRCast.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"
#include "../SRPlatform/Interface/SRCpuInfo.h"
#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRDefaultLogger.h"
#include "../SRPlatform/Interface/SRDoubleNumber.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRFastArray.h"
#include "../SRPlatform/Interface/SRFastRandom.h"
#include "../SRPlatform/Interface/SRFinally.h"
#include "../SRPlatform/Interface/SRHeap.h"
#include "../SRPlatform/Interface/SRLambdaSubtask.h"
#include "../SRPlatform/Interface/SRLock.h"
#include "../SRPlatform/Interface/SRLogStream.h"
#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRMaxSizeof.h"
#include "../SRPlatform/Interface/SRMemPool.h"
#include "../SRPlatform/Interface/SRMinimalTask.h"
#include "../SRPlatform/Interface/SRPoolRunner.h"
#include "../SRPlatform/Interface/SRReaderWriterSync.h"
#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRSpinSync.h"
#include "../SRPlatform/Interface/SRTaskWaiter.h"
#include "../SRPlatform/Interface/SRThreadPool.h"
#include "../SRPlatform/Interface/SRUtils.h"
#include "../SRPlatform/Interface/SRVectMath.h"
