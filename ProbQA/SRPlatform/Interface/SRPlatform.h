// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#ifdef SRPLATFORM_EXPORTS
#define SRPLATFORM_API __declspec(dllexport)
#else
#define SRPLATFORM_API __declspec(dllimport)
#endif // SRPLATFORM_EXPORTS

//// IS_CPU_X86_32 , IS_CPU_X86_64
#ifdef _MSC_VER
  #if _M_IX86
    static_assert(sizeof(void*) == 4, "Double-checking for CPU bit-width detection");
    #define IS_CPU_X86_32 1
    #define IS_CPU_X86_64 0
  #elif _M_X64
    static_assert(sizeof(void*) == 8, "Double-checking for CPU bit-width detection");
    #define IS_CPU_X86_32 0
    #define IS_CPU_X86_64 1
  #else
    #error This CPU is not supported yet.
  #endif /* CPU selection under MSVC++ compiler */
#else
  #error This compiler is not supported yet.
#endif /* Compiler selection */

#if IS_CPU_X86_32
inline int _rdrand64_step(unsigned __int64* val) {
  uint32_t *p = reinterpret_cast<uint32_t*>(val);
  return _rdrand32_step(p) & _rdrand32_step(p+1);
}
#endif /* IS_CPU_X86_32 */
