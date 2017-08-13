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
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#pragma warning( pop )
