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

// STL
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <tchar.h>
#include <thread>
#include <vector>

// SRPlatform library includes
#include "../SRPlatform/Interface/SRCast.h"
#include "../SRPlatform/Interface/SRFinally.h"
#include "../SRPlatform/Interface/SRFastRandom.h"
#include "../SRPlatform/Interface/SRMath.h"
