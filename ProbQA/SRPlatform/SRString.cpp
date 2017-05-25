#include "stdafx.h"
#include "../SRPlatform/Interface/SRString.h"

namespace SRPlat {

SRString::SRString(const char *const pData, const bool bOwned, const size_t length)
  : _pData(pData), _bOwned(bOwned ? 1 : 0) {
  if (length == std::string::npos) {
    _length = strlen(pData);
  }
  else {
    _length = length;
  }
}

SRString::~SRString() {
  if (_bOwned) {
    delete _pData;
  }
}

SRString SRString::MakeOwned(const char *const pData, size_t length) {
  return SRString(pData, true, length);
}

SRString SRString::MakeClone(const char *const pData, size_t length) {
  if (length == std::string::npos) {
    length = strlen(pData);
  }
  char* const pDup = new char[length];
  memcpy(pDup, pData, length);
  return SRString(pDup, true, length);
}

SRString SRString::MakeUnowned(const char *const pData, size_t length) {
  return SRString(pData, false, length);
}

} // namespace SRPlat