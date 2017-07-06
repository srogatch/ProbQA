// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRString.h"

namespace SRPlat {

class SRMessageBuilder {
  std::string _buf;
public:
  explicit SRMessageBuilder() {}
  explicit SRMessageBuilder(const char* const pInit) : _buf(pInit) { }
  
  SRMessageBuilder& operator()(const char* const pS) {
    _buf.append(pS);
    return *this;
  }

  template<typename T> SRMessageBuilder& operator()(const T& val) {
    _buf += std::to_string(val);
    return *this;
  }

  template<> SRMessageBuilder& operator()(const std::string& s) {
    _buf.append(s);
    return *this;
  }

  template<> SRMessageBuilder& operator()(const SRString& srs) {
    const char* pData;
    size_t len = srs.GetData(pData);
    _buf.append(pData, len);
    return *this;
  }

  SRMessageBuilder& operator()(const void* const ptr) {
    char buf[16];
    snprintf(buf, sizeof(buf), "0x%p", ptr);
    _buf.append(buf);
    return *this;
  }

  // operator() may treat it as arithmetic type
  SRMessageBuilder& AppendChar(char ch) {
    _buf += ch;
    return *this;
  }

  // Get the string suitable only for the current module (EXE/DLL)
  const std::string& GetString() {
    return _buf;
  }

  // Get the string suitable for passing between modules
  SRString GetOwnedSRString() {
    return SRString::MakeClone(_buf.c_str(), _buf.size());
  }

  // Get the string usable until modification or destruction of this builder.
  SRString GetUnownedSRString() {
    return SRString::MakeUnowned(_buf.c_str(), _buf.size());
  }
};

} // namespace SRPlat
