#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRString.h"

namespace SRPlat {

class SRMessageBuilder {
  std::stringstream _ss;
public:
  explicit SRMessageBuilder() {}
  
  template<typename T> SRMessageBuilder& operator()(const T& arg) {
    _ss << arg;
    return *this;
  }

  const char* GetStr() {
    return _ss.str().c_str();
  }

  std::string GetString() {
    return _ss.str();
  }

  SRString GetOwnedSRString() {
    std::string s = _ss.str();
    return SRString::MakeClone(s.c_str(), s.size());
  }
};

} // namespace SRPlat