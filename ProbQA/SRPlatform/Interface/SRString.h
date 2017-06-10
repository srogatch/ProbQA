#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {

// A string for DLL/EXE interoperability. The class is not intended for string modification, but rather just for passing
//   constant string between modules.
//NOTE: the string may not contain a null terminator.
//TODO: why is this final?
class SRPLATFORM_API SRString final {
  const char *_pData;
  uint64_t _length : 63; // let it occupy bits 0..62 so to avoid a shift operation on access
  uint64_t _bOwned : 1;

private: // methods
  SRString(const char *const pData, const bool bOwned, const size_t length);
  static char* DupData(const char* const pData, const size_t length);
  void setSameData(const char* const pData);

public: // methods
  SRString();
  SRString(const SRString& fellow);
  SRString& operator=(const SRString& fellow);
  SRString(SRString&& fellow);
  SRString& operator=(SRString&& fellow);
  ~SRString();

  explicit SRString(const std::string& source);

  static SRString MakeOwned(const char *const pData, size_t length = std::string::npos);
  static SRString MakeClone(const char *const pData, size_t length = std::string::npos);
  static SRString MakeUnowned(const char *const pData, size_t length = std::string::npos);

  // Must be inline, in order to use std::string from the same module.
  std::string ToString() const {
    return std::string(_pData, _length);
  }

  // Returns the length of the data. Stores the pointer in the output parameter. Note that data may not be
  //   null-terminated.
  size_t GetData(const char* &outData) const;
};

} // namespace SRPlat
