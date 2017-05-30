#pragma once

namespace SRPlat {

template<uint32_t taYieldPeriod> class SRSpinSync {
  std::atomic_flag _af = ATOMIC_FLAG_INIT;
public:
  void Acquire() {
    uint32_t nSpins = 0;
    while (_af.test_and_set(std::memory_order_acquire)) {
      nSpins++;
      if (nSpins >= taYieldPeriod) {
        nSpins = 0;
        std::this_thread::yield();
      }
    }
  }
  void Release() {
    _af.clear(std::memory_order_release);
  }
};

} // namespace SRPlat
