// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "targetver.h"

#include <Windows.h>
#undef min
#undef max

#include <immintrin.h>

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

#include "../SRPlatform/Interface/SRFinally.h"
#include "../SRPlatform/Interface/SRFastRandom.h"
