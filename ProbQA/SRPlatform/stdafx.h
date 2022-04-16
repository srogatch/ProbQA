// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#ifdef _WIN32
  #define _CRT_SECURE_NO_WARNINGS
  #include "targetver.h"
  #define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
  // Windows Header Files:
  #include <windows.h>
  #undef min
  #undef max
#endif // _WIN32

// CPU-specific header files
#include <immintrin.h>

#if defined(_WIN32)
  #include <intrin.h>
#elif defined(__unix__)
  #include <x86intrin.h>
#endif // OS

#ifdef _WIN32
  #pragma warning( push )
  #pragma warning( disable : 4251 ) // needs to have dll-interface to be used by clients of class
#endif // _WIN32

// STL
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

#if defined(_WIN32)
  #include <io.h>
#elif defined(__unix__)
  #include <inttypes.h>
  #include <unistd.h>
  #define __int64 int64_t
  #define _close close
  #define _read read
  #define _lseek64 lseek64
  #define _O_RDONLY O_RDONLY
  #define _open open
  #define _lseeki64 lseek64
  #define _lseek lseek
  #define stricmp strcasecmp
#endif // OS

#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>

#ifdef _WIN32
  #pragma warning( pop )
#endif // _WIN32
