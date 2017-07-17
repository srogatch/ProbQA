// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {

class SRPLATFORM_API SRSpinStatistics {
  template <uint32_t YP> friend class SRSpinSync;

private: // methods
  // Returns the total number of spins together with the current spin
  static uint64_t OnContention();

public: // methods
  static uint64_t TotalContention();
};

template<uint32_t taYieldPeriod> class SRPLATFORM_API SRSpinSync {
  std::atomic_flag _af = ATOMIC_FLAG_INIT;
public:
  void Acquire() {
    uint32_t nSpins = 0;
    while (_af.test_and_set(std::memory_order_acquire)) {
      SRSpinStatistics::OnContention();
      nSpins++;
      if (nSpins >= taYieldPeriod) {
        nSpins = 0;
        std::this_thread::yield();
      }
      else {
        // From https://software.intel.com/sites/landingpage/IntrinsicsGuide/ :
        // "Provide a hint to the processor that the code sequence is a spin-wait loop. This can help improve the
        //   performance and power consumption of spin-wait loops." 
        _mm_pause();
      }
    }
  }
  void Release() {
    _af.clear(std::memory_order_release);
  }
};

} // namespace SRPlat
