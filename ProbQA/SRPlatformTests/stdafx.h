// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <Windows.h>
#undef min
#undef max

// CPU-specific header files
#include <immintrin.h>
#include <intrin.h>

// STL
#pragma warning( push )
#pragma warning( disable : 4251 ) // needs to have dll-interface to be used by clients of class
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <tchar.h>
#include <thread>
#include <vector>
#pragma warning( pop )

// SRPlatform library includes
#include "../SRPlatform/Interface/SRAccumVectDbl256.h"
#include "../SRPlatform/Interface/SRBitArray.h"
#include "../SRPlatform/Interface/SRBucketSummatorPar.h"
#include "../SRPlatform/Interface/SRBucketSummatorSeq.h"
#include "../SRPlatform/Interface/SRFastRandom.h"
#include "../SRPlatform/Interface/SRHeap.h"
#include "../SRPlatform/Interface/SRQueue.h"
#include "../SRPlatform/Interface/SRVectMath.h"

// Google Test Framework includes
#include <gtest/gtest.h>
