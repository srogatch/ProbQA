#pragma once

#include "../SRPlatform/Interface/SRMacros.h"

namespace SRPlat {

template <typename taFunc> class SRFinally {
  taFunc _f;
public:
  SRFinally(taFunc&& f) : _f(std::forward<taFunc>(f)) {}
  ~SRFinally() { _f(); }
  SRFinally(const SRFinally&) = delete;
  SRFinally& operator=(const SRFinally&) = delete;
  SRFinally(SRFinally&&) = delete;
  SRFinally& operator=(SRFinally&&) = delete;
};

#define SR_FINALLY(destructorVar)                                                                         \
  auto SR_COMBINE(finallyFunc, __LINE) = destructorVar;                                                   \
  ::SRPlat::SRFinally<decltype(SR_COMBINE(finallyFunc, __LINE))> SR_COMBINE(finallyObj, __LINE__) (       \
    std::move(SR_COMBINE(finallyFunc, __LINE)));
}