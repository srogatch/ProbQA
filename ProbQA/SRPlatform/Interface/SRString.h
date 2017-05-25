#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {

// A string for DLL/EXE interoperability. The class is not intended for string modification, but rather just for passing
//   constant string between modules.
//NOTE: the string may not contain a null terminator.
class SRPLATFORM_API SRString final {
  const char *_pData;
  uint64_t _bOwned : 1;
  uint64_t _length : 63;

private: // methods
  SRString(const char *const pData, const bool bOwned, const size_t length);
  static char* DupData(const char* const pData, const size_t length);
  void setSameData(const char* const pData);

public: // methods
  SRString(const SRString& fellow);
  SRString& operator=(const SRString& fellow);
  SRString(SRString&& fellow);
  SRString& operator=(SRString&& fellow);
  ~SRString();

  static SRString MakeOwned(const char *const pData, size_t length = std::string::npos);
  static SRString MakeClone(const char *const pData, size_t length = std::string::npos);
  static SRString MakeUnowned(const char *const pData, size_t length = std::string::npos);

  std::string ToString();
};

} // namespace SRPlat
