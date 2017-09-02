// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRMacros.h"

namespace SRPlat {

// A string for DLL/EXE interoperability. The class is not intended for string modification, but rather just for passing
//   constant string between modules.
//NOTE: the string may not contain a null terminator.
// It is final to avoid any need for a virtual destructor.
class SRPLATFORM_API SRString final {
  const char *_pData;
  uint64_t _length : 63; // let it occupy bits 0..62 so to avoid a shift operation on access
  uint64_t _bOwned : 1;

private: // methods
  SRString(const char *PTR_RESTRICT const pData, const bool bOwned, const size_t length);
  ATTR_RESTRICT static char* DupData(const char *PTR_RESTRICT const pData, const size_t length);
  void setSameData(const char *PTR_RESTRICT const pData);

public: // methods
  SRString();
  SRString(const SRString& fellow);
  SRString& operator=(const SRString& fellow);
  SRString(SRString&& fellow);
  SRString& operator=(SRString&& fellow);
  ~SRString();

  explicit SRString(const std::string& PTR_RESTRICT source);

  static SRString MakeOwned(const char *PTR_RESTRICT const pData, size_t length = std::string::npos);
  static SRString MakeClone(const char *PTR_RESTRICT const pData, size_t length = std::string::npos);
  static SRString MakeUnowned(const char *PTR_RESTRICT const pData, size_t length = std::string::npos);

  // Must be inline, in order to use std::string from the same module.
  std::string ToString() const {
    return std::string(_pData, _length);
  }

  // Returns the length of the data. Stores the pointer in the output parameter. Note that data may not be
  //   null-terminated.
  size_t GetData(const char *PTR_RESTRICT &PTR_RESTRICT outData) const;
};

} // namespace SRPlat
