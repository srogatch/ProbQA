#include "stdafx.h"
#include "../SRPlatform/Interface/SRString.h"

namespace SRPlat {

char* SRString::DupData(const char* const pData, const size_t length) {
  char* const pDup = new char[length];
  memcpy(pDup, pData, length);
  return pDup;
}

void SRString::setSameData(const char* const pData) {
  if (_bOwned) {
    _pData = DupData(pData, _length);
  }
  else {
    _pData = pData;
  }
}

SRString::SRString(const char *const pData, const bool bOwned, const size_t length)
  : _pData(pData), _bOwned(bOwned ? 1 : 0) {
  if (length == std::string::npos) {
    _length = strlen(pData);
  }
  else {
    _length = length;
  }
}

SRString::SRString(const SRString& fellow) {
  _length = fellow._length;
  _bOwned = fellow._bOwned;
  setSameData(fellow._pData);
}

SRString& SRString::operator=(const SRString& fellow) {
  if (this != &fellow) {
    if (_bOwned) {
      delete _pData;
    }
    _length = fellow._length;
    _bOwned = fellow._bOwned;
    setSameData(fellow._pData);
  }
  return *this;
}

SRString::SRString(SRString&& fellow) : _pData(fellow._pData), _bOwned(fellow._bOwned), _length(fellow._length) {
  fellow._pData = nullptr;
  fellow._bOwned = false;
  fellow._length = 0;
}

SRString& SRString::operator=(SRString&& fellow) {
  if (this != &fellow) {
    this->_pData = fellow._pData;
    this->_bOwned = fellow._bOwned;
    this->_length = fellow._length;
    fellow._pData = nullptr;
    fellow._bOwned = false;
    fellow._length = 0;
  }
  return *this;
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
  return SRString(DupData(pData, length), true, length);
}

SRString SRString::MakeUnowned(const char *const pData, size_t length) {
  return SRString(pData, false, length);
}

std::string SRString::ToString() {
  return std::string(_pData, _length);
}

} // namespace SRPlat