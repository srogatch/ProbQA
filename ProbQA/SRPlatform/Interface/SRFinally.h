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

template <typename Func> SRFinally<Func> SRMakeFinally(Func&& f) {
  return { std::forward<Func>(f) };
}

#define SR_FINALLY(funcVar) \
  __pragma(warning(push)) \
  __pragma(warning(disable: 4189)) \
  auto&& SR_COMBINE(srFinally, __LINE__) = ::SRPlat::SRMakeFinally(funcVar); \
  __pragma(warning(pop))

}