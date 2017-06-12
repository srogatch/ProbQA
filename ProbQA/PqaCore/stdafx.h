// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>

#include <immintrin.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRAlignedAllocator.h"
#include "../SRPlatform/Interface/SRSpinSync.h"
#include "../SRPlatform/Interface/SRLock.h"
#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"
#include "../SRPlatform/Interface/SRReaderWriterSync.h"
#include "../SRPlatform/Interface/ISRLogger.h"
#include "../SRPlatform/Interface/SRDefaultLogger.h"
#include "../SRPlatform/Interface/SRLogStream.h"
#include "../SRPlatform/Interface/SRMemPool.h"

// TODO: reference additional headers your program requires here
