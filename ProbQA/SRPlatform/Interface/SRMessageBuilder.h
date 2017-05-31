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

  //NOTE: don't do the following, because temporary string gets deallocated:
  //  return _ss.str().c_str();

  // Get the string suitable only for the current module (EXE/DLL)
  std::string GetString() {
    return _ss.str();
  }

  // Get the string suitable for passing between modules
  SRString GetOwnedSRString() {
    std::string s = _ss.str();
    return SRString::MakeClone(s.c_str(), s.size());
  }
};

} // namespace SRPlat
