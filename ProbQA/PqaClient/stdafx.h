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
#include <cinttypes>
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
#include "../SRPlatform/Interface/ISRLogger.h"
#include "../SRPlatform/Interface/SRBasicTypes.h"
#include "../SRPlatform/Interface/SRBitArray.h"
#include "../SRPlatform/Interface/SRCast.h"
#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRDefaultLogger.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRFastArray.h"
#include "../SRPlatform/Interface/SRFastRandom.h"
#include "../SRPlatform/Interface/SRFinally.h"
#include "../SRPlatform/Interface/SRLock.h"
#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRMemPool.h"
#include "../SRPlatform/Interface/SRMPAllocator.h"
#include "../SRPlatform/Interface/SRString.h"

// PqaCore library includes
#include "../PqaCore/Interface/IPqaEngineFactory.h"
