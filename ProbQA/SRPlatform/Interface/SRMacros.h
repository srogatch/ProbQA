// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

// Disable "conditional expression is constant" warning in `do { } while(false);` loops
#define WHILE_FALSE               \
  __pragma(warning(push))         \
  __pragma(warning(disable:4127)) \
  while(false)                    \
  __pragma(warning(pop))

#define SR_STRINGIZE(x) SR_STRINGIZE2(x)
#define SR_STRINGIZE2(x) #x
#define SR_LINE_STRING SR_STRINGIZE(__LINE__)

#define SR_COMBINE2(x, y) x ## y
#define SR_COMBINE(x, y) SR_COMBINE2(x, y)

#define SR_FILE_LINE __FILE__ "(" SR_LINE_STRING "): "

#define ATTR_NOALIAS __declspec(noalias)

#if defined(_WIN32)
#define ATTR_RESTRICT __declspec(restrict)
#elif defined(__unix__)
#define ATTR_RESTRICT __restrict__
#else
#error "Unhandled OS"
#endif // OS

#define ATTR_NORETURN __declspec(noreturn)
#define ATTR_NOINLINE __declspec(noinline)
#define ATTR_NOVTABLE __declspec(novtable)

#define PTR_RESTRICT __restrict
#define SR_UNREACHABLE __assume(0)

#define FLOAT_PRECISE_BEGIN __pragma(float_control(push)) __pragma(float_control(precise, on))
#define FLOAT_PRECISE_END __pragma(float_control(pop))

// Cast for *printf format
#define CASTF_HU(var) static_cast<unsigned short>(var)
#define CASTF_DU(var) static_cast<unsigned int>(var)

#define SR_STACK_ALLOC(typeVar, countVar) static_cast<typeVar*>(_alloca(sizeof(typeVar) * (countVar)))
// _alloca() alignment seems 16 bytes: https://docs.microsoft.com/en-us/cpp/build/alloca . Thus we need no more than
//  16 padding bytes to make it 32-byte aligned.
#define SR_ALIGNED_ALLOCA_PADDING 16
#define SR_STACK_ALLOC_ALIGN(typeVar, countVar) static_cast<typeVar*>( \
  SRSimd::AlignPtr(_alloca(sizeof(typeVar) * (countVar) + SR_ALIGNED_ALLOCA_PADDING), SR_ALIGNED_ALLOCA_PADDING))
